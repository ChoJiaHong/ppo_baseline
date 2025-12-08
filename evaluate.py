"""
Evaluation script for trained PPO model.
"""
import argparse
import numpy as np
import torch

from environment import ServiceDeploymentEnv
from policy import ActorCritic


def evaluate(args):
    """Evaluate a trained policy."""
    
    # Create environment
    env = ServiceDeploymentEnv(
        num_nodes=args.num_nodes,
        num_services=args.num_services,
        max_agents=args.max_agents
    )
    
    # Create policy network
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print("Loaded checkpoint from:", args.checkpoint_path)
    print("-" * 80)
    
    # Evaluation loop
    episode_rewards = []
    
    for episode in range(args.num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        if args.render:
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}")
            print(f"{'='*50}")
        
        while not done:
            # Select action (deterministic)
            action, _, _ = policy.act(state)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            if args.render:
                env.render()
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        if not args.render:
            print(f"Episode {episode + 1}/{args.num_episodes} - Reward: {episode_reward:.2f}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("=" * 80)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model')
    
    # Environment parameters
    parser.add_argument('--num_nodes', type=int, default=3, help='Number of nodes')
    parser.add_argument('--num_services', type=int, default=5, help='Number of services')
    parser.add_argument('--max_agents', type=int, default=10, help='Maximum number of agents')
    
    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    
    # Evaluation parameters
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/ppo_final.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render environment during evaluation')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()
