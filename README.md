# PPO Baseline - Reinforcement Learning for Service Deployment

A complete baseline implementation of **PPO (Proximal Policy Optimization)** for multi-service deployment and resource allocation problems.

## 🎯 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Demo
```bash
python demo.py
```

### Train Model
```bash
python train.py
```

### Evaluate Model
```bash
python evaluate.py --checkpoint_path checkpoints/ppo_final.pt
```

## 📚 Documentation

- **[README_PPO.md](README_PPO.md)** - Complete usage guide (Chinese/中文)
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details (Chinese/中文)
- **[config.yaml](config.yaml)** - Configuration file with all hyperparameters

## 🏗️ Architecture

### Core Components

1. **Environment** (`environment.py`)
   - Gymnasium-based custom environment
   - Handles 5 event types: agent arrival/departure, node failure/recovery, no event
   - State space: event type, node status, occupancy, agent count
   - Action space: deploy to node 0..N or reject

2. **Policy Network** (`policy.py`)
   - Actor-Critic architecture
   - Actor: outputs action probability distribution
   - Critic: outputs state value function
   - Shared feature extraction layers

3. **Rollout Buffer** (`buffer.py`)
   - Stores training trajectories
   - Implements GAE (Generalized Advantage Estimation)
   - Computes returns and advantages

4. **PPO Trainer** (`ppo_trainer.py`)
   - Complete PPO algorithm implementation
   - Clipped surrogate objective
   - Value function clipping
   - Entropy regularization
   - Gradient clipping

### Scripts

- **train.py** - Main training script with TensorBoard integration
- **evaluate.py** - Evaluation script for trained models
- **demo.py** - Quick demonstration for easy onboarding

## 🔧 Key Features

- ✅ Standard PPO algorithm with all modern improvements
- ✅ Configurable hyperparameters via command line or config file
- ✅ TensorBoard integration for monitoring
- ✅ Automatic checkpoint saving
- ✅ Comprehensive documentation in Chinese
- ✅ Ready-to-use demo script
- ✅ No security vulnerabilities (CodeQL verified)

## 📊 Algorithm Details

### PPO Components
- **Clipped Surrogate Objective**: Prevents too large policy updates
- **GAE**: Balances bias and variance in advantage estimation
- **Multiple Epochs**: Reuses collected data efficiently
- **Entropy Regularization**: Encourages exploration
- **Gradient Clipping**: Ensures training stability

### Default Hyperparameters
- Learning rate: 3e-4
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Clip epsilon (ε): 0.2
- Value coefficient: 0.5
- Entropy coefficient: 0.01
- Update epochs: 4
- Batch size: 64

## 🚀 Usage Examples

### Basic Training
```bash
python train.py --total_timesteps 100000
```

### Custom Parameters
```bash
python train.py --num_nodes 5 --lr 0.0001 --total_timesteps 200000
```

### Monitor with TensorBoard
```bash
tensorboard --logdir runs/
```

### Evaluate with Rendering
```bash
python evaluate.py --render --num_episodes 5
```

## 📈 Expected Results

After successful training, the agent should be able to:
- Effectively deploy services to available nodes
- Avoid deploying to failed nodes
- Maximize successful deployments within node capacity constraints
- Adapt to dynamic events (agent arrivals/departures, node failures/recoveries)

## 🔍 Project Statistics

- **Total Python Code**: ~960 lines
- **Files**: 9 Python modules + documentation
- **Dependencies**: PyTorch, Gymnasium, NumPy, Matplotlib, TensorBoard
- **Security**: 0 vulnerabilities (CodeQL verified)

## 🛠️ Troubleshooting

### Training not converging
- Lower learning rate: `--lr 1e-4`
- Increase update interval: `--update_interval 4096`
- Adjust entropy coefficient: `--entropy_coef 0.02`

### Out of memory
- Reduce batch size: `--batch_size 32`
- Reduce update interval: `--update_interval 1024`

### Training too slow
- Reduce total timesteps: `--total_timesteps 50000`
- Reduce update epochs: `--update_epochs 2`

## 📖 References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 🔮 Future Extensions

As mentioned in the project documentation, this baseline can be extended with:
1. W table integration (throughput constraints)
2. Constrained RL (Lagrangian methods)
3. Feasible action projection
4. More complex reward shaping
5. Larger-scale environments

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Issues and Pull Requests are welcome!

---

**Status**: ✅ Complete and tested  
**Language**: Python 3.12+  
**Framework**: PyTorch 2.0+  
**Environment**: Gymnasium
