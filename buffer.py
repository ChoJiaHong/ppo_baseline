"""
Rollout buffer for storing trajectories.
"""
import numpy as np
import torch
from typing import List, Tuple


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO training.
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            log_prob: float, value: float, done: bool):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        """Clear all stored data."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all data as tensors.
        
        Returns:
            states, actions, rewards, log_probs, values as tensors
        """
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        log_probs = torch.FloatTensor(self.log_probs)
        values = torch.FloatTensor(self.values)
        
        return states, actions, rewards, log_probs, values
    
    def compute_returns_and_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            returns: Discounted returns
            advantages: Computed advantages
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)
    
    def __len__(self) -> int:
        """Return the number of transitions stored."""
        return len(self.states)
