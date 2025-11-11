# Mountain-Car-RL

Reinforcement Learning project implementing Proximal Policy Optimization (PPO) to solve the Mountain Car environment from Gymnasium.

## 🚗 Project Overview

This project trains an AI agent to solve the classic Mountain Car problem, where an underpowered car must learn to build momentum and reach a goal position on top of a hill.

## ✨ Features

- **PPO Algorithm Implementation** using Stable Baselines3
- **Real-time Training Visualization** - Watch the car learn in real-time
- **Interactive Dashboards** - Comprehensive HTML visualizations
- **Bilingual Documentation** - English and Greek (Ελληνικά)
- **Complete Training Pipeline** - From training to evaluation
- **TensorBoard Integration** - Monitor training progress

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train with Visualization

```bash
python train_with_visualization.py
```

This will:
- Train a PPO agent for 500,000 timesteps
- Show the car learning every 50 episodes
- Save metrics automatically
- Generate visualizations at the end

### 3. View Results

```bash
# Generate interactive dashboards
python visualize_training.py

# Open in browser
start .\visualizations\main_dashboard.html

# Watch trained agent
python view_agent.py
```

## 📁 Project Structure

```
Mountain Car RL Algorithm/
├── mountain_car_ppo.py          # Main training script
├── train_with_visualization.py  # Training with real-time visualization
├── visualize_training.py        # Interactive dashboard generator
├── view_agent.py                # Watch trained agent play
├── check_training_status.py     # Diagnostic tool
├── extract_tensorboard_metrics.py # Extract metrics from TensorBoard
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── README_EL.md                 # Greek version
├── REPORT.md                    # Comprehensive project report
├── REPORT_EL.md                 # Greek report
├── VISUALIZATION_GUIDE.md       # Visualization guide
├── VISUALIZATION_GUIDE_EL.md    # Greek guide
├── ΟΔΗΓΙΕΣ_ΕΚΤΕΛΕΣΗΣ_EL.md     # Execution guide (Greek)
├── models/                      # Trained models
├── results/                     # Training results and plots
├── visualizations/              # Interactive HTML dashboards
├── training_logs/               # Training metrics
└── tensorboard_logs/            # TensorBoard logs
```

## 🎯 Environment Details

- **Environment**: Mountain Car v0 (Gymnasium)
- **Observation Space**: Box(2,) - [position, velocity]
- **Action Space**: Discrete(3) - [left, no acceleration, right]
- **Reward**: -1 per timestep (sparse rewards)
- **Goal**: Reach position ≥ 0.5

## 🧠 Algorithm

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: Stable Baselines3
- **Policy**: Multi-layer Perceptron (MlpPolicy)

## 📊 Results

After training, the agent should achieve:
- **Success Rate**: >80%
- **Mean Reward**: -100 to -150
- **Mean Episode Length**: 100-150 steps

## 📚 Documentation

- **English**: See `REPORT.md` and `VISUALIZATION_GUIDE.md`
- **Ελληνικά**: Δείτε `REPORT_EL.md` και `VISUALIZATION_GUIDE_EL.md`

## 🛠️ Usage Examples

### Train without visualization (faster)
```bash
python mountain_car_ppo.py
```

### Train with more frequent visualization
```bash
python train_with_visualization.py --freq 25
```

### Check training status
```bash
python check_training_status.py
```

### Extract metrics from TensorBoard
```bash
python extract_tensorboard_metrics.py
```

## 📈 Monitoring

- **TensorBoard**: `tensorboard --logdir ./tensorboard_logs/`
- **Interactive Dashboards**: Open `./visualizations/main_dashboard.html`
- **Training Metrics**: Automatically saved to `./training_logs/`

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available for educational purposes.

## 🔗 Links

- [Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/mountain_car/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

---

**Enjoy training your Mountain Car agent! 🚗⛰️**
