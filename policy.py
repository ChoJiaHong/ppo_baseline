"""
PPO Policy Network (Actor-Critic).
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Tuple


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs action probabilities
    Critic: Outputs state value
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, obs_dim)
            
        Returns:
            action_probs: Action probabilities of shape (batch_size, action_dim)
            value: State value of shape (batch_size, 1)
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select an action given state.
        
        Args:
            state: State observation
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: State value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = self.forward(state_tensor)
        
        # Sample action from probability distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given states.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Entropy of action distribution
        """
        action_probs, values = self.forward(states)
        
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy
