"""
Main training script for PPO baseline.
"""
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from environment import ServiceDeploymentEnv
from policy import ActorCritic
from ppo_trainer import PPOTrainer
from buffer import RolloutBuffer


def train(args):
    """Main training loop."""
    
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
    
    # Create PPO trainer
    trainer = PPOTrainer(
        policy=policy,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
    )
    
    # Create buffer
    buffer = RolloutBuffer()
    
    # Create tensorboard writer
    log_dir = os.path.join('runs', f'ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    writer = SummaryWriter(log_dir)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    
    # Main training loop
    print("Starting PPO training...")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Update interval: {args.update_interval}")
    print("-" * 80)
    
    global_step = 0
    episode_num = 0
    
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    while global_step < args.total_timesteps:
        # Collect rollout
        for _ in range(args.update_interval):
            # Select action
            action, log_prob, value = policy.act(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.add(state, action, reward, log_prob, value, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1
            
            # Handle episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_num += 1
                
                # Log episode statistics
                writer.add_scalar('Episode/Reward', episode_reward, episode_num)
                writer.add_scalar('Episode/Length', episode_length, episode_num)
                
                if episode_num % args.log_interval == 0:
                    avg_reward = np.mean(episode_rewards[-args.log_interval:])
                    avg_length = np.mean(episode_lengths[-args.log_interval:])
                    print(f"Episode {episode_num} | Steps: {global_step} | "
                          f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f}")
                
                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            
            if global_step >= args.total_timesteps:
                break
        
        # Update policy
        if len(buffer) > 0:
            stats = trainer.update(buffer)
            buffer.clear()
            
            # Log training statistics
            writer.add_scalar('Train/PolicyLoss', stats['policy_loss'], global_step)
            writer.add_scalar('Train/ValueLoss', stats['value_loss'], global_step)
            writer.add_scalar('Train/Entropy', stats['entropy'], global_step)
        
        # Save checkpoint
        if global_step % args.save_interval == 0 and global_step > 0:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f'checkpoints/ppo_checkpoint_{global_step}.pt'
            trainer.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final save
    os.makedirs('checkpoints', exist_ok=True)
    final_path = 'checkpoints/ppo_final.pt'
    trainer.save(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary:")
    print(f"Total episodes: {episode_num}")
    print(f"Total timesteps: {global_step}")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train PPO on service deployment environment')
    
    # Environment parameters
    parser.add_argument('--num_nodes', type=int, default=3, help='Number of nodes')
    parser.add_argument('--num_services', type=int, default=5, help='Number of services')
    parser.add_argument('--max_agents', type=int, default=10, help='Maximum number of agents')
    
    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    
    # PPO parameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--update_epochs', type=int, default=4, help='Number of update epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total timesteps')
    parser.add_argument('--update_interval', type=int, default=2048, help='Steps between updates')
    parser.add_argument('--log_interval', type=int, default=10, help='Episodes between logging')
    parser.add_argument('--save_interval', type=int, default=10000, help='Steps between checkpoints')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
