# Interactive Visualization Guide

This guide explains how to use the interactive visualization tools for the Mountain Car PPO project.

## 📊 Available Visualizations

### 1. **Training Dashboard** (`visualize_training.py`)

Creates interactive HTML dashboards showing:
- **Training Progress**: Real-time episode rewards and lengths
- **Hyperparameters**: Model configuration visualization
- **Agent Performance**: How the agent performs in episodes
- **Summary Dashboard**: Comprehensive statistics

### 2. **Real-time Agent Viewer** (`view_agent.py`)

Watch the trained agent play Mountain Car in real-time with a visual window.

## 🚀 Quick Start

### Step 1: Train the Model (if not already done)

```bash
python mountain_car_ppo.py
```

This will:
- Train the PPO agent
- Save training metrics to `./training_logs/training_metrics.json`
- Automatically generate visualizations at the end

### Step 2: Generate Interactive Dashboards

After training completes, run:

```bash
python visualize_training.py
```

This creates HTML files in the `./visualizations/` folder:
- `main_dashboard.html` - Main entry point with links to all dashboards
- `training_dashboard.html` - Training progress and learning curves
- `hyperparameters.html` - Model hyperparameters visualization
- `agent_performance.html` - Agent behavior analysis
- `summary_dashboard.html` - Comprehensive summary

### Step 3: View the Dashboards

Open `./visualizations/main_dashboard.html` in your web browser to access all interactive visualizations!

**Features:**
- ✨ Interactive plots (zoom, pan, hover for details)
- 📈 Real-time data visualization
- 🎯 Performance metrics
- ⚙️ Hyperparameter display

### Step 4: Watch the Agent Play

To see the agent in action:

```bash
python view_agent.py
```

**Options:**
```bash
# View 5 episodes (default)
python view_agent.py

# View specific number of episodes
python view_agent.py ./models/mountain_car_ppo_final.zip 10

# Control playback speed (seconds per step)
python view_agent.py ./models/mountain_car_ppo_final.zip 5 0.1
```

## 📁 File Structure

```
Mountain Car RL Algorithm/
├── mountain_car_ppo.py          # Main training script (with monitoring)
├── visualize_training.py        # Interactive dashboard generator
├── view_agent.py                # Real-time agent viewer
├── training_logs/
│   └── training_metrics.json   # Training metrics (auto-generated)
├── visualizations/
│   ├── main_dashboard.html      # Main dashboard entry point
│   ├── training_dashboard.html  # Training progress
│   ├── hyperparameters.html    # Hyperparameters
│   ├── agent_performance.html  # Agent performance
│   └── summary_dashboard.html  # Summary statistics
└── models/
    └── mountain_car_ppo_final.zip  # Trained model
```

## 🎨 Dashboard Features

### Training Dashboard
- **Episode Rewards**: Scatter plot showing reward per episode
- **Episode Lengths**: How long each episode lasted
- **Mean Rewards**: 10-episode rolling average
- **Mean Lengths**: 10-episode rolling average
- **Interactive**: Hover to see exact values, zoom and pan

### Hyperparameters Dashboard
- Visual bar chart of all PPO hyperparameters
- Shows learning rate, batch size, discount factor, etc.
- Color-coded for easy reading

### Agent Performance Dashboard
- **Position Over Time**: How the car's position changes
- **Velocity Over Time**: Speed changes during episode
- **Phase Space**: Position vs Velocity trajectory
- **Actions Over Time**: Which actions the agent chose
- Shows multiple episodes for comparison

### Summary Dashboard
- Training progress curve
- Evaluation results (box plots)
- Reward distribution histogram
- Episode length distribution
- Success rate statistics

## 🔄 Real-time Monitoring

During training, metrics are automatically saved to `./training_logs/training_metrics.json`. You can:

1. **Monitor during training**: The metrics file updates every 10 episodes
2. **Generate dashboards anytime**: Run `visualize_training.py` even while training is in progress
3. **View in browser**: Open the HTML files - they update when you refresh

## 💡 Tips

1. **Best Browser**: Use Chrome, Firefox, or Edge for best compatibility
2. **Refresh Dashboards**: If training is ongoing, refresh the HTML to see latest data
3. **Multiple Episodes**: View agent with more episodes to see consistency
4. **TensorBoard**: Also check `tensorboard --logdir ./tensorboard_logs/` for additional metrics

## 🐛 Troubleshooting

**No visualizations generated?**
- Make sure training completed successfully
- Check that `./training_logs/training_metrics.json` exists
- Verify `plotly` and `pandas` are installed: `pip install plotly pandas`

**Agent viewer not working?**
- Ensure model file exists: `./models/mountain_car_ppo_final.zip`
- Check that `gymnasium` is installed with rendering support
- On some systems, you may need: `pip install pyglet`

**Dashboards not interactive?**
- Make sure you're opening the HTML files in a web browser (not a text editor)
- Check browser console for any JavaScript errors
- Try a different browser

## 📊 Example Usage Workflow

```bash
# 1. Train the model
python mountain_car_ppo.py

# 2. Generate visualizations (auto-done after training, or run manually)
python visualize_training.py

# 3. Open main dashboard in browser
# Windows: start visualizations/main_dashboard.html
# Mac: open visualizations/main_dashboard.html
# Linux: xdg-open visualizations/main_dashboard.html

# 4. Watch agent play
python view_agent.py
```

Enjoy exploring your Mountain Car PPO training results! 🚗⛰️

