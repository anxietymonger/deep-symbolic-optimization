from abc import ABC, abstractmethod

import torch

from dso.program import Program
from dso.memory import Batch
from dso.utils import import_custom_source
from dso.policy.policy import Policy


class PolicyOptimizer(ABC):
    """Abstract class for a policy optimizer using PyTorch."""

    def _init(self,
            policy: Policy,
            debug: int = 0,
            summary: bool = False,
            optimizer: str = 'adam',
            learning_rate: float = 0.001,
            entropy_weight: float = 0.005,
            entropy_gamma: float = 1.0) -> None:

        self.policy = policy
        self.debug = debug
        self.summary = summary
        self.n_choices = Program.library.L
        self.entropy_weight = entropy_weight
        self.entropy_gamma = entropy_gamma

        # Setup optimizer
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        if self.debug >= 1:
            total_parameters = sum(p.numel() for p in self.policy.parameters())
            print("\nModel parameters:")
            for name, param in self.policy.named_parameters():
                print(f"Variable: {name}")
                print(f"  Shape: {param.shape}")
                print(f"  Parameters: {param.numel()}")
            print(f"Total parameters: {total_parameters}")

    @abstractmethod
    def train_step(self, baseline: torch.Tensor, sampled_batch: Batch) -> dict:
        """Returns metrics dictionary instead of TF summaries"""
        raise NotImplementedError


def make_policy_optimizer(policy, policy_optimizer_type, **config_policy_optimizer):
    """Factory function for policy optimizer object."""

    if policy_optimizer_type == "pg":
        from dso.policy_optimizer.pg_policy_optimizer import PGPolicyOptimizer
        policy_optimizer_class = PGPolicyOptimizer
    else:
        policy_optimizer_class = import_custom_source(policy_optimizer_type)
        assert issubclass(policy_optimizer_class, Policy), \
                f"Custom policy {policy_optimizer_class} must subclass dso.policy.Policy."

    policy_optimizer = policy_optimizer_class(policy, **config_policy_optimizer)
    return policy_optimizer
