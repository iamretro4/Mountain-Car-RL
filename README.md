# Mountain Car RL Algorithm - PPO Implementation

This project implements a Proximal Policy Optimization (PPO) algorithm to solve the Mountain Car environment from Gymnasium.

## Project Structure

```
Mountain Car RL Algorithm/
├── mountain_car_ppo.py    # Main training script
├── requirements.txt       # Python dependencies
├── REPORT.md             # Comprehensive project report
└── README.md             # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
python mountain_car_ppo.py
```

This will:
- Train a PPO agent for 500,000 timesteps
- Save the trained model to `./models/`
- Generate evaluation results and visualizations in `./results/`

### 3. View Results

After training, check:
- `./models/mountain_car_ppo_final.zip` - Final trained model
- `./results/training_analysis.png` - Training visualizations
- `./results/phase_space.png` - Phase space plot
- `./tensorboard_logs/` - TensorBoard logs (view with `tensorboard --logdir ./tensorboard_logs/`)

## Environment Details

- **Environment**: Mountain Car v0 (Gymnasium)
- **Observation Space**: Box(2,) - [position, velocity]
- **Action Space**: Discrete(3) - [left, no acceleration, right]
- **Reward**: -1 per timestep (sparse rewards)
- **Goal**: Reach position ≥ 0.5

## Algorithm

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: Stable Baselines3
- **Policy**: Multi-layer Perceptron (MlpPolicy)

## Documentation

See `REPORT.md` for:
- Detailed environment description
- Algorithm selection rationale
- Code explanations
- Comparison with other RL algorithms
- Results analysis

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Stable Baselines3
- NumPy
- Matplotlib

## Notes

- Training may take 30-60 minutes depending on your hardware
- The model will be saved periodically during training
- Use TensorBoard to monitor training progress in real-time

# Mountain-Car-RL
