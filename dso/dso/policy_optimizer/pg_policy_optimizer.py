import torch

from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy


class PGPolicyOptimizer(PolicyOptimizer):
    """Vanilla policy gradient policy optimizer using PyTorch."""

    def __init__(self,
            policy: Policy,
            debug: int = 0,
            summary: bool = False,
            optimizer: str = 'adam',
            learning_rate: float = 0.001,
            entropy_weight: float = 0.005,
            entropy_gamma: float = 1.0) -> None:

        super()._init(policy, debug, summary, optimizer, learning_rate,
                     entropy_weight, entropy_gamma)

    def train_step(self, baseline, sampled_batch):
        """Computes loss, trains model, and returns metrics."""
        # Convert numpy arrays to tensors if needed
        if not isinstance(baseline, torch.Tensor):
            baseline = torch.tensor(baseline, dtype=torch.float32)

        # Get neglogp and entropy from policy
        neglogp, entropy = self.policy.make_neglogp_and_entropy(sampled_batch, self.entropy_gamma)

        # Compute rewards
        rewards = torch.tensor(sampled_batch.rewards, dtype=torch.float32)

        # Calculate losses
        pg_loss = ((rewards - baseline) * neglogp).mean()
        entropy_loss = -self.entropy_weight * entropy.mean()
        total_loss = pg_loss + entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Return metrics dictionary instead of TF summaries
        metrics = {
            "pg_loss": pg_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "baseline": baseline.item(),
            "grad_norm": grad_norm.item()
        }

        return metrics
