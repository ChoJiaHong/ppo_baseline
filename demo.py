"""
Quick demo script to showcase PPO training.
This is a minimal example to get started quickly.
"""
from environment import ServiceDeploymentEnv
from policy import ActorCritic
from ppo_trainer import PPOTrainer
from buffer import RolloutBuffer
import numpy as np


def quick_demo():
    """Run a quick demonstration of PPO training."""
    print("=" * 80)
    print("PPO Training Quick Demo")
    print("=" * 80)
    
    # Create environment
    env = ServiceDeploymentEnv(num_nodes=3, num_services=5, max_agents=10)
    print(f"\n✓ Created environment with {env.num_nodes} nodes")
    
    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)
    print(f"✓ Created Actor-Critic policy (obs_dim={obs_dim}, action_dim={action_dim})")
    
    # Create trainer
    trainer = PPOTrainer(policy, lr=3e-4)
    print(f"✓ Created PPO trainer")
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting training for 10 episodes...")
    print(f"{'='*80}\n")
    
    buffer = RolloutBuffer()
    episode_rewards = []
    
    for episode in range(10):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob, value = policy.act(state)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.add(state, action, reward, log_prob, value, done)
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d} | Reward: {episode_reward:6.2f}")
        
        # Update policy every episode (for demo purposes)
        if len(buffer) > 0:
            trainer.update(buffer)
            buffer.clear()
    
    # Summary
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"{'='*80}\n")
    
    # Save model
    trainer.save('demo_model.pt')
    print("✓ Model saved to 'demo_model.pt'")
    
    # Test the trained model
    print("\nTesting trained model for 3 episodes:")
    print("-" * 80)
    
    policy.eval()
    test_rewards = []
    
    for episode in range(3):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _, _ = policy.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1} | Reward: {episode_reward:.2f}")
    
    print(f"\nTest average reward: {np.mean(test_rewards):.2f}")
    print("=" * 80)
    
    env.close()


if __name__ == '__main__':
    quick_demo()
