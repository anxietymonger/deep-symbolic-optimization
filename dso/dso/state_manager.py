from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from dso.program import Program


class StateManager(ABC):
    """
    An interface for handling the torch.Tensor inputs to the Policy.
    """

    def setup_manager(self, policy):
        """
        Function called inside the policy to perform the needed initializations
        :param policy the policy class
        """
        self.policy = policy
        self.max_length = policy.max_length

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Convert an observation from a Task into a Tensor input for the
        Policy, e.g. by performing one-hot encoding or embedding lookup.

        Parameters
        ----------
        obs : np.ndarray (dtype=np.float32)
            Observation coming from the Task.

        Returns
        --------
        input_ : torch.Tensor (dtype=torch.float32)
            Tensor to be used as input to the Policy.
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs


def make_state_manager(config):
    """
    Parameters
    ----------
    config : dict
        Parameters for this StateManager.

    Returns
    -------
    state_manager : StateManager
        The StateManager to be used by the policy.
    """
    manager_dict = {
        "hierarchical": HierarchicalStateManager
    }

    if config is None:
        config = {}

    # Use HierarchicalStateManager by default
    manager_type = config.pop("type", "hierarchical")

    manager_class = manager_dict[manager_type]
    state_manager = manager_class(**config)

    return state_manager


class HierarchicalStateManager(StateManager):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self, observe_parent=True, observe_sibling=True,
                 observe_action=False, observe_dangling=False, embedding=False,
                 embedding_size=8):
        super().__init__()
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.library = Program.library

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size
        self.embedding_layers = nn.ModuleDict()

    def setup_manager(self, policy):
        super().setup_manager(policy)
        # Create embeddings if needed
        if self.embedding:
            if self.observe_action:
                self.embedding_layers["action"] = nn.Embedding(
                    self.library.n_action_inputs, self.embedding_size)
            if self.observe_parent:
                self.embedding_layers["parent"] = nn.Embedding(
                    self.library.n_parent_inputs, self.embedding_size)
            if self.observe_sibling:
                self.embedding_layers["sibling"] = nn.Embedding(
                    self.library.n_sibling_inputs, self.embedding_size)

            # Initialize embeddings with uniform distribution
            for embedding in self.embedding_layers.values():
                nn.init.uniform_(embedding.weight, -1.0, 1.0)

    def get_tensor_input(self, obs):
        observations = []
        # Convert obs to PyTorch tensor if it's not already
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        # Unstack along dimension 1
        unstacked_obs = torch.unbind(obs, dim=1)
        action, parent, sibling, dangling = unstacked_obs[:4]

        # Cast to int32 instead of long (int64)
        action = action.to(torch.int64)
        parent = parent.to(torch.int64)
        sibling = sibling.to(torch.int64)

        if self.observe_action:
            if self.embedding:
                x = self.embedding_layers["action"](action)
            else:
                x = torch.nn.functional.one_hot(action, self.library.n_action_inputs).to(torch.float32)
            observations.append(x)

        if self.observe_parent:
            if self.embedding:
                x = self.embedding_layers["parent"](parent)
            else:
                x = torch.nn.functional.one_hot(parent, self.library.n_parent_inputs).to(torch.float32)
            observations.append(x)

        if self.observe_sibling:
            if self.embedding:
                x = self.embedding_layers["sibling"](sibling)
            else:
                x = torch.nn.functional.one_hot(sibling, self.library.n_sibling_inputs).to(torch.float32)
            observations.append(x)

        if self.observe_dangling:
            x = dangling.unsqueeze(-1)
            observations.append(x)

        input_ = torch.cat(observations, dim=-1)
        # Concatenate additional observations if they exist
        if len(unstacked_obs) > 4:
            additional_obs = torch.stack(unstacked_obs[4:], dim=-1)
            input_ = torch.cat([input_, additional_obs], dim=-1)

        return input_
