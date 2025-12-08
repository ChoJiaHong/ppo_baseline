"""
PPO (Proximal Policy Optimization) Trainer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional

from policy import ActorCritic
from buffer import RolloutBuffer


class PPOTrainer:
    """
    PPO training algorithm.
    
    Key features:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus for exploration
    """
    
    def __init__(
        self,
        policy: ActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy: Actor-Critic policy network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            update_epochs: Number of epochs for policy update
            batch_size: Mini-batch size for updates
        """
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Update policy using PPO algorithm.
        
        Args:
            buffer: Rollout buffer with collected trajectories
            
        Returns:
            Dictionary with training statistics
        """
        # Get data from buffer
        states, actions, rewards, old_log_probs, old_values = buffer.get()
        
        # Compute returns and advantages
        returns, advantages = buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # Multiple epochs of updates
        for epoch in range(self.update_epochs):
            # Mini-batch updates
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Compute policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss (clipped)
                values_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss1 = (values - batch_returns).pow(2)
                value_loss2 = (values_clipped - batch_returns).pow(2)
                value_loss = torch.max(value_loss1, value_loss2).mean()
                
                # Compute entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Return training statistics
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
        
        return stats
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
