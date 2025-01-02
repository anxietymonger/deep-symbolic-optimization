"""Controller used to generate distribution over hierarchical, variable-length objects."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dso.program import Program
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
        # Get input size from state manager's processed state
        dummy_obs = torch.zeros(1, Program.task.OBS_DIM, device=self.device)
        processed_obs = self.state_manager.process_state(dummy_obs)
        input_size = self.state_manager.get_tensor_input(processed_obs).size(-1)

        # Create recurrent cell
        if cell == "lstm":
            rnn = nn.LSTM(input_size=input_size, hidden_size=num_units, num_layers=num_layers, batch_first=True)
        elif cell == "gru":
            rnn = nn.GRU(input_size=input_size, hidden_size=num_units, num_layers=num_layers, batch_first=True)
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

        input = self.state_manager.get_tensor_input(B.obs)
        logits, _ = self.rnn(input, hidden)
        if self.action_prob_lowerbound != 0.0:
            logits = self.apply_action_prob_lowerbound(logits)

        logits += B.priors
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
        B_max_length = B.actions.size(1)
        mask = torch.arange(B_max_length, device=self.device).expand(len(B.lengths), B_max_length) < torch.Tensor(B.lengths).unsqueeze(1)
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
        """Sample a batch of n expressions."""
        self.eval()
        with torch.no_grad():
            actions = []
            obs = []
            priors = []
            hidden = None
            next_obs = None
            next_prior = None
            finished = np.zeros(n, dtype=bool)
            for t in range(self.max_length):
                if t == 0:
                    initial_obs = Program.task.reset_task(self.prior)
                    initial_obs = torch.tensor(initial_obs, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
                    initial_obs = self.state_manager.process_state(initial_obs)
                    initial_prior = torch.tensor(self.prior.initial_prior(), dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)

                    ob = initial_obs
                    prior = initial_prior
                else:
                    next_obs, next_prior, finished = Program.task.get_next_obs(
                        torch.stack(actions, dim=1).cpu().numpy(),
                        ob.cpu().numpy(),
                        finished
                    )
                    ob = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    prior = torch.tensor(next_prior, dtype=torch.float32, device=self.device)

                input = self.state_manager.get_tensor_input(ob)
                logits, hidden = self.rnn(input.unsqueeze(1), hidden)
                logits = logits.squeeze(1)

                if self.action_prob_lowerbound != 0.0:
                    logits = self.apply_action_prob_lowerbound(logits)

                logits += prior
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(1)
                actions.append(action)
                obs.append(ob)
                priors.append(prior)

                if finished.all():
                    break

            actions = torch.stack(actions, dim=1)
            obs = torch.stack(obs, dim=2)
            priors = torch.stack(priors, dim=1)

        return actions, obs, priors

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
