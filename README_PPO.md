# PPO Baseline 基礎強化學習代碼

這是一個基於 PPO (Proximal Policy Optimization) 算法的強化學習基礎代碼，用於解決多服務部署與資源分配問題。

## 📋 專案結構

```
ppo_baseline/
├── environment.py      # 服務部署環境 (Gymnasium)
├── policy.py          # Actor-Critic 策略網路
├── buffer.py          # 經驗回放緩衝區
├── ppo_trainer.py     # PPO 訓練器
├── train.py           # 主訓練腳本
├── evaluate.py        # 評估腳本
├── requirements.txt   # 依賴套件
└── README_PPO.md      # 本文檔
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 訓練模型

使用預設參數訓練：

```bash
python train.py
```

自訂參數訓練：

```bash
python train.py --total_timesteps 200000 --num_nodes 5 --lr 0.0003
```

### 3. 評估模型

```bash
python evaluate.py --checkpoint_path checkpoints/ppo_final.pt --num_episodes 10
```

顯示詳細過程：

```bash
python evaluate.py --checkpoint_path checkpoints/ppo_final.pt --render
```

## 🎯 環境說明

### 狀態空間 (Observation Space)
- 事件類型 (one-hot 編碼)
  - agent_arrival: 新代理加入
  - agent_departure: 代理離開
  - node_failure: 節點故障
  - node_recovery: 節點恢復
  - no_event: 無事件
- 節點狀態 (活躍/故障)
- 節點佔用率 (正規化)
- 當前代理數量 (正規化)

### 動作空間 (Action Space)
- 0 到 N-1: 部署到節點 0 至 N-1
- N: 拒絕部署

### 獎勵函數 (Reward Function)
- +1.0: 成功部署服務
- -0.5: 拒絕部署
- -1.0: 部署失敗 (節點故障或容量不足)
- -2.0 × 服務數: 節點故障導致服務丟失

## 🔧 PPO 算法參數

### 核心參數
- `lr`: 學習率 (預設: 3e-4)
- `gamma`: 折扣因子 (預設: 0.99)
- `gae_lambda`: GAE λ 參數 (預設: 0.95)
- `clip_epsilon`: PPO 裁剪參數 (預設: 0.2)
- `value_coef`: 價值損失係數 (預設: 0.5)
- `entropy_coef`: 熵獎勵係數 (預設: 0.01)

### 訓練參數
- `total_timesteps`: 總訓練步數 (預設: 100000)
- `update_interval`: 更新間隔 (預設: 2048)
- `update_epochs`: 每次更新的訓練輪數 (預設: 4)
- `batch_size`: 小批次大小 (預設: 64)

## 📊 監控訓練

使用 TensorBoard 監控訓練過程：

```bash
tensorboard --logdir runs/
```

然後在瀏覽器打開 `http://localhost:6006`

### 追蹤的指標
- Episode Reward: 每回合的總獎勵
- Episode Length: 每回合的步數
- Policy Loss: 策略損失
- Value Loss: 價值損失
- Entropy: 策略熵 (探索程度)

## 💡 使用範例

### 基本訓練
```python
from environment import ServiceDeploymentEnv
from policy import ActorCritic
from ppo_trainer import PPOTrainer
from buffer import RolloutBuffer

# 創建環境
env = ServiceDeploymentEnv(num_nodes=3, num_services=5, max_agents=10)

# 創建策略網路
policy = ActorCritic(obs_dim=env.observation_space.shape[0], 
                     action_dim=env.action_space.n)

# 創建訓練器
trainer = PPOTrainer(policy)

# 訓練循環
buffer = RolloutBuffer()
for step in range(total_steps):
    action, log_prob, value = policy.act(state)
    next_state, reward, done, _, _ = env.step(action)
    buffer.add(state, action, reward, log_prob, value, done)
    
    if step % update_interval == 0:
        trainer.update(buffer)
        buffer.clear()
```

## 📈 預期結果

訓練成功後，代理應該能夠：
1. 有效地將服務部署到可用節點
2. 避免部署到故障節點
3. 在節點容量限制下最大化成功部署
4. 適應動態事件 (代理加入/離開、節點故障/恢復)

## 🔍 故障排除

### 訓練不收斂
- 降低學習率 `--lr 1e-4`
- 增加更新間隔 `--update_interval 4096`
- 調整熵係數 `--entropy_coef 0.02`

### 記憶體不足
- 減少批次大小 `--batch_size 32`
- 減少更新間隔 `--update_interval 1024`

### 訓練太慢
- 減少總步數 `--total_timesteps 50000`
- 減少更新輪數 `--update_epochs 2`

## 📚 參考資料

- [PPO 原始論文](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Gymnasium 文檔](https://gymnasium.farama.org/)

## 📝 授權

此專案為教育與研究用途。

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！
