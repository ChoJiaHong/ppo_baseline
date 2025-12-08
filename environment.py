"""
Simple service deployment environment for PPO baseline.
Based on the problem described in readme.md.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from typing import Dict, List, Tuple, Optional


class ServiceDeploymentEnv(gym.Env):
    """
    A simplified service deployment environment.
    
    The agent must decide where to deploy services across multiple nodes,
    considering throughput constraints and node failures.
    
    Events:
    - agent_arrival: A new agent requests service
    - agent_departure: An agent leaves
    - node_failure: A node fails
    - node_recovery: A node recovers
    - no_event: Nothing happens
    
    Actions:
    - Deploy to node 1, 2, 3, ... , N
    - Reject/No action
    """
    
    def __init__(self, num_nodes: int = 3, num_services: int = 5, max_agents: int = 10,
                 node_capacity: float = 5.0, event_probabilities: Optional[List[float]] = None):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.max_agents = max_agents
        self.default_node_capacity = node_capacity
        self.event_probabilities = event_probabilities or [0.3, 0.2, 0.1, 0.1, 0.3]
        
        # Event types
        self.EVENT_TYPES = ['agent_arrival', 'agent_departure', 'node_failure', 'node_recovery', 'no_event']
        self.num_event_types = len(self.EVENT_TYPES)
        
        # Action space: deploy to node 0, 1, ..., N-1, or reject (N)
        self.action_space = spaces.Discrete(num_nodes + 1)
        
        # Observation space:
        # - event type (one-hot): num_event_types
        # - node status (binary): num_nodes
        # - node occupancy (number of services): num_nodes
        # - agent count: 1
        obs_dim = self.num_event_types + self.num_nodes + self.num_nodes + 1
        self.observation_space = spaces.Box(
            low=0, high=float(self.max_agents), shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Node status: 1 = active, 0 = failed
        self.node_status = np.ones(self.num_nodes, dtype=np.float32)
        
        # Node occupancy: number of services deployed on each node
        self.node_occupancy = np.zeros(self.num_nodes, dtype=np.float32)
        
        # Max capacity per node
        self.node_capacity = np.full(self.num_nodes, self.default_node_capacity, dtype=np.float32)
        
        # Current number of active agents
        self.num_agents = 0
        
        # Current event
        self.current_event = 'no_event'
        self.current_event_idx = self.EVENT_TYPES.index(self.current_event)
        
        # Step counter
        self.steps = 0
        self.max_steps = 100
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Event type one-hot
        event_onehot = np.zeros(self.num_event_types, dtype=np.float32)
        event_onehot[self.current_event_idx] = 1.0
        
        obs = np.concatenate([
            event_onehot,
            self.node_status,
            self.node_occupancy / self.node_capacity,  # Normalized occupancy
            np.array([self.num_agents / self.max_agents], dtype=np.float32)
        ])
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Node index to deploy (0 to num_nodes-1) or num_nodes for reject
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        info = {}
        
        # Process action based on current event
        if self.current_event == 'agent_arrival':
            if action < self.num_nodes:
                # Try to deploy to selected node
                node_idx = action
                if self.node_status[node_idx] == 1.0 and self.node_occupancy[node_idx] < self.node_capacity[node_idx]:
                    # Successful deployment
                    self.node_occupancy[node_idx] += 1.0
                    self.num_agents += 1
                    reward = 1.0
                    info['deployment'] = 'success'
                else:
                    # Failed deployment (node down or full)
                    reward = -1.0
                    info['deployment'] = 'failed'
            else:
                # Reject action
                reward = -0.5
                info['deployment'] = 'rejected'
        
        elif self.current_event == 'agent_departure':
            if self.num_agents > 0:
                # Remove agent from a random node
                occupied_nodes = np.where(self.node_occupancy > 0)[0]
                if len(occupied_nodes) > 0:
                    node_idx = self.np_random.choice(occupied_nodes)
                    self.node_occupancy[node_idx] -= 1.0
                    self.num_agents -= 1
                    reward = 0.0
        
        elif self.current_event == 'node_failure':
            # Random node fails
            active_nodes = np.where(self.node_status == 1.0)[0]
            if len(active_nodes) > 0:
                failed_node = self.np_random.choice(active_nodes)
                self.node_status[failed_node] = 0.0
                # Lose all services on that node
                lost_services = self.node_occupancy[failed_node]
                self.node_occupancy[failed_node] = 0.0
                self.num_agents = max(0, int(self.num_agents - lost_services))
                reward = -2.0 * lost_services
                info['node_failure'] = failed_node
        
        elif self.current_event == 'node_recovery':
            # Random failed node recovers
            failed_nodes = np.where(self.node_status == 0.0)[0]
            if len(failed_nodes) > 0:
                recovered_node = self.np_random.choice(failed_nodes)
                self.node_status[recovered_node] = 1.0
                reward = 0.5
                info['node_recovery'] = recovered_node
        
        # Generate next event
        self.steps += 1
        self._generate_next_event()
        
        # Check termination
        terminated = self.steps >= self.max_steps
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _generate_next_event(self):
        """Generate the next random event."""
        # Event probabilities: arrival, departure, failure, recovery, no_event
        self.current_event_idx = self.np_random.choice(len(self.EVENT_TYPES), p=self.event_probabilities)
        self.current_event = self.EVENT_TYPES[self.current_event_idx]
    
    def render(self):
        """Print current state."""
        print(f"Step: {self.steps}")
        print(f"Event: {self.current_event}")
        print(f"Active agents: {self.num_agents}")
        print(f"Node status: {self.node_status}")
        print(f"Node occupancy: {self.node_occupancy}")
        print("-" * 50)
