"""Controller used to generate distribution over hierarchical, variable-length objects."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dso.program import Program
from dso.program import _finish_tokens
from dso.policy import Policy


class LinearWrapper(nn.Module):
    """RNN wrapper that adds a linear layer to the output."""

    def __init__(self, rnn_cell, output_size):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.linear = nn.Linear(rnn_cell.hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn_cell(x, hidden)
        return self.linear(output), hidden

def safe_cross_entropy(p, logq, dim=-1):
    """Compute p * logq safely."""
    # Handle cases where p is 0
    safe_logq = torch.where(p == 0, torch.ones_like(logq), logq)
    return -torch.sum(p * safe_logq, dim=dim)

class RNNPolicy(Policy):
    def __init__(self, prior, state_manager,
                 debug=0,
                 max_length=30,
                 action_prob_lowerbound=0.0,
                 max_attempts_at_novel_batch=10,
                 sample_novel_batch=False,
                 cell="lstm",
                 num_layers=1,
                 num_units=32,
                 initializer="zeros"):
        super().__init__(prior, state_manager, debug, max_length)

        self.action_prob_lowerbound = action_prob_lowerbound
        self.n_choices = Program.library.L
        self.max_attempts_at_novel_batch = max_attempts_at_novel_batch
        self.sample_novel_batch = sample_novel_batch

        # Move to PyTorch device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self._setup_model(cell, num_layers, num_units, initializer)
        self.to(self.device)

    def _setup_model(self, cell="lstm", num_layers=1, num_units=32, initializer="zeros"):
        if isinstance(num_units, int):
            num_units = [num_units] * num_layers

        # Get input size from state manager's processed state
        dummy_obs = torch.zeros(1, Program.task.OBS_DIM, device=self.device)
        processed_obs = self.state_manager.process_state(dummy_obs)
        input_size = self.state_manager.get_tensor_input(processed_obs).size(-1)

        # Create recurrent cell
        if cell == "lstm":
            rnn = nn.LSTM(input_size=input_size,
                         hidden_size=num_units[0],
                         num_layers=num_layers,
                         batch_first=True)
        elif cell == "gru":
            rnn = nn.GRU(input_size=input_size,
                        hidden_size=num_units[0],
                        num_layers=num_layers,
                        batch_first=True)
        else:
            raise ValueError(f"Unsupported cell type: {cell}")

        self.rnn = LinearWrapper(rnn, self.n_choices)

        # Initialize weights
        if initializer == "zeros":
            for p in self.parameters():
                if len(p.shape) > 1:
                    nn.init.zeros_(p)
        elif initializer == "var_scale":
            for p in self.parameters():
                if len(p.shape) > 1:
                    nn.init.kaiming_uniform_(p, a=np.sqrt(5))
        else:
            raise ValueError(f"Unsupported initializer: {initializer}")

    def make_neglogp_and_entropy(self, B, entropy_gamma):
        """Computes the negative log-probabilities for a given
        batch of actions, observations and priors
        under the current policy.

        Returns
        -------
        neglogp, entropy :
            PyTorch tensors
        """
        if entropy_gamma is None:
            entropy_gamma = 1.0
        entropy_gamma_decay = torch.tensor([entropy_gamma**t for t in range(self.max_length)], dtype=torch.float32, device=self.device)

        # Initialize hidden state
        batch_size = B.obs.size(0)
        if isinstance(self.rnn.rnn_cell, nn.LSTM):
            h0 = torch.zeros(self.rnn.rnn_cell.num_layers, batch_size, self.rnn.rnn_cell.hidden_size, device=self.device)
            c0 = torch.zeros(self.rnn.rnn_cell.num_layers, batch_size, self.rnn.rnn_cell.hidden_size, device=self.device)
            hidden = (h0, c0)
        else:  # GRU
            hidden = torch.zeros(self.rnn.rnn_cell.num_layers, batch_size, self.rnn.rnn_cell.hidden_size, device=self.device)

        logits, _ = self.rnn(B.obs, hidden)
        if self.action_prob_lowerbound != 0.0:
            logits = self.apply_action_prob_lowerbound(logits)

        logits += B.priors
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
        B_max_length = B.actions.size(1)
        mask = torch.arange(B_max_length, device=self.device).expand(len(B.lengths), B_max_length) < B.lengths.unsqueeze(1)
        mask = mask.float()

        actions_one_hot = F.one_hot(B.actions, num_classes=self.n_choices).float()
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, dim=2)
        neglogp = torch.sum(neglogp_per_step * mask, dim=1)

        sliced_entropy_gamma_decay = entropy_gamma_decay[:B_max_length]
        entropy_gamma_decay_mask = sliced_entropy_gamma_decay * mask
        entropy_per_step = safe_cross_entropy(probs, logprobs, dim=2)
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay_mask, dim=1)

        return neglogp, entropy

    def sample(self, n: int):
        """Sample batch of n expressions

        Returns
        -------
        actions, obs, priors :
            Or a batch
        """
        if self.sample_novel_batch:
            actions, obs, priors = self.sample_novel(n)
        else:
            actions, obs, priors = self._sample(n)

        return actions, obs, priors

    def _sample(self, n: int):
        """Sample a batch of n expressions."""
        self.eval()
        with torch.no_grad():
            batch_size = torch.tensor(n, device=self.device)
            initial_obs = Program.task.reset_task(self.prior)
            initial_obs = torch.tensor(initial_obs, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
            initial_obs = self.state_manager.process_state(initial_obs)

            initial_prior = torch.tensor(self.prior.initial_prior(), dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)

            actions = []
            obs = []
            priors = []

            hidden = None
            all_actions = []  # Track all actions for proper shape
            for t in range(self.max_length):
                if t == 0:
                    input = self.state_manager.get_tensor_input(initial_obs)
                    prior = initial_prior
                else:
                    input = self.state_manager.get_tensor_input(next_obs)
                    prior = next_prior

                logits, hidden = self.rnn(input.unsqueeze(1), hidden)
                logits = logits.squeeze(1)

                if self.action_prob_lowerbound != 0.0:
                    logits = self.apply_action_prob_lowerbound(logits)

                logits += prior
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(1)

                actions.append(action)
                obs.append(input)
                priors.append(prior)

                # Create proper action history for get_next_obs
                all_actions.append(action)
                actions_history = torch.stack(all_actions, dim=1)  # [batch_size, t+1]

                next_obs, next_prior, finished = Program.task.get_next_obs(
                    actions_history.cpu().numpy(),
                    input.cpu().numpy(),
                    np.zeros(n, dtype=bool)
                )
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                next_prior = torch.tensor(next_prior, dtype=torch.float32, device=self.device)

                if finished.all():
                    break

            actions = torch.stack(actions, dim=1)
            obs = torch.stack(obs, dim=2)
            priors = torch.stack(priors, dim=1)

            pad_length = self.max_length - actions.size(1)
            if pad_length > 0:
                actions = F.pad(actions, (0, pad_length), value=0)
                obs = F.pad(obs, (0, pad_length), value=0)
                priors = F.pad(priors, (0, pad_length, 0, 0), value=0)

        return actions, obs, priors

    def sample_novel(self, n: int):
        """Sample a batch of n expressions not contained in cache.

        If unable to do so within self.max_attempts_at_novel_batch,
        then fills in the remaining slots with previously-seen samples.

        Parameters
        ----------
        n: int
            batch size

        Returns
        -------
        unique_a, unique_o, unique_p: np.ndarrays
        """
        n_novel = 0
        old_a, old_o, old_p = [], [], []
        new_a, new_o, new_p = [], [], []
        n_attempts = 0
        while n_novel < n and n_attempts < self.max_attempts_at_novel_batch:
            actions, obs, priors = self._sample(n)
            n_attempts += 1
            new_indices = []
            old_indices = []
            for idx, a in enumerate(actions):
                # Convert tensor to numpy before passing to _finish_tokens
                tokens = a.cpu().numpy()
                tokens = _finish_tokens(tokens)
                key = tokens.tostring()  # Use numpy's tostring() directly
                if key not in Program.cache.keys() and n_novel < n:
                    new_indices.append(idx)
                    n_novel += 1
                if key in Program.cache.keys():
                    old_indices.append(idx)

            new_a.append(actions[new_indices])
            new_o.append(obs[new_indices])
            new_p.append(priors[new_indices])
            old_a.append(actions[old_indices])
            old_o.append(obs[old_indices])
            old_p.append(priors[old_indices])

        n_remaining = n - n_novel

        # Pad everything to max_length
        for tensors, dim in [(old_a, 1), (new_a, 1),
                           (old_o, 2), (new_o, 2),
                           (old_p, 1), (new_p, 1)]:
            if tensors:  # Only pad if there are tensors in the list
                max_length = max(t.size(dim) for t in tensors)
                tensors[:] = self._pad_batch(tensors, dim, max_length)

        # Concatenate padded tensors
        old_a = torch.cat(old_a) if old_a else torch.empty(0, device=self.device)
        old_o = torch.cat(old_o) if old_o else torch.empty(0, device=self.device)
        old_p = torch.cat(old_p) if old_p else torch.empty(0, device=self.device)

        # Include redundant samples if needed
        new_a = torch.cat(new_a + [old_a[:n_remaining]])
        new_o = torch.cat(new_o + [old_o[:n_remaining]])
        new_p = torch.cat(new_p + [old_p[:n_remaining]])

        # Store extended batch for later use
        self.extended_batch = [old_a.size(0), old_a, old_o, old_p]
        self.valid_extended_batch = True

        return new_a, new_o, new_p

    def _pad_batch(self, tensors, dim, max_length=None, pad_value=0):
        """Pad a list of tensors to the same length along specified dimension."""
        if max_length is None:
            max_length = max(t.size(dim) for t in tensors)

        padded_tensors = []
        for t in tensors:
            pad_size = max_length - t.size(dim)
            if pad_size > 0:
                pad_shape = list(t.shape)
                pad_shape[dim] = pad_size
                padding = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
                t = torch.cat([t, padding], dim=dim)
            padded_tensors.append(t)
        return padded_tensors

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""
        self.eval()
        with torch.no_grad():
            if log:
                fetch = self.memory_logps
            else:
                fetch = self.memory_probs
            probs = fetch(memory_batch)
        return probs

    def apply_action_prob_lowerbound(self, logits):
        """Applies a lower bound to probabilities of each action.

        Parameters
        ----------
        logits: torch.Tensor where last dimension has size self.n_choices

        Returns
        -------
        logits_bounded: torch.Tensor
        """
        probs = F.softmax(logits, dim=-1)
        probs_bounded = ((1 - self.action_prob_lowerbound) * probs +
                         self.action_prob_lowerbound / float(self.n_choices))
        logits_bounded = torch.log(probs_bounded)

        return logits_bounded
