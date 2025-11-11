"""
Interactive Visualization Dashboard for Mountain Car PPO Training
Shows training progress, parameters, and results
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import os
import json
from datetime import datetime
import time

# ============================================================================
# Training Monitor Callback
# ============================================================================

class TrainingMonitor(BaseCallback):
    """
    Callback to monitor training progress and save metrics.
    """
    def __init__(self, log_dir="./training_logs/", verbose=0):
        super(TrainingMonitor, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.mean_rewards = []
        self.mean_lengths = []
        
    def _on_step(self) -> bool:
        # Get info from the environment
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            if 'r' in episode_info:
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
                self.timesteps.append(self.num_timesteps)
                
                # Calculate running means
                if len(self.episode_rewards) >= 10:
                    self.mean_rewards.append(np.mean(self.episode_rewards[-10:]))
                    self.mean_lengths.append(np.mean(self.episode_lengths[-10:]))
                else:
                    self.mean_rewards.append(np.mean(self.episode_rewards))
                    self.mean_lengths.append(np.mean(self.episode_lengths))
                
                # Save metrics periodically
                if len(self.episode_rewards) % 10 == 0:
                    self.save_metrics()
        
        return True
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics = {
            'timesteps': self.timesteps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_rewards': self.mean_rewards,
            'mean_lengths': self.mean_lengths
        }
        with open(os.path.join(self.log_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f)

# ============================================================================
# Interactive Training Dashboard
# ============================================================================

def create_training_dashboard(metrics_file="./training_logs/training_metrics.json"):
    """
    Create an interactive Plotly dashboard showing training progress.
    """
    if not os.path.exists(metrics_file):
        print("No training metrics found. Run training first.")
        return None
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    timesteps = metrics['timesteps']
    episode_rewards = metrics['episode_rewards']
    episode_lengths = metrics['episode_lengths']
    mean_rewards = metrics['mean_rewards']
    mean_lengths = metrics['mean_lengths']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Episode Lengths', 
                       'Mean Reward (10-episode average)', 'Mean Length (10-episode average)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Episode Rewards
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=episode_rewards,
            mode='markers',
            name='Episode Reward',
            marker=dict(size=4, opacity=0.6, color='blue'),
            hovertemplate='Timestep: %{x}<br>Reward: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_hline(y=-200, line_dash="dash", line_color="red", 
                  annotation_text="Max Episode Length", row=1, col=1)
    
    # Plot 2: Episode Lengths
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=episode_lengths,
            mode='markers',
            name='Episode Length',
            marker=dict(size=4, opacity=0.6, color='green'),
            hovertemplate='Timestep: %{x}<br>Length: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Plot 3: Mean Rewards
    if mean_rewards:
        fig.add_trace(
            go.Scatter(
                x=timesteps[:len(mean_rewards)],
                y=mean_rewards,
                mode='lines',
                name='Mean Reward',
                line=dict(color='orange', width=2),
                hovertemplate='Timestep: %{x}<br>Mean Reward: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Plot 4: Mean Lengths
    if mean_lengths:
        fig.add_trace(
            go.Scatter(
                x=timesteps[:len(mean_lengths)],
                y=mean_lengths,
                mode='lines',
                name='Mean Length',
                line=dict(color='purple', width=2),
                hovertemplate='Timestep: %{x}<br>Mean Length: %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Timesteps", row=2, col=1)
    fig.update_xaxes(title_text="Timesteps", row=2, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Length", row=1, col=2)
    fig.update_yaxes(title_text="Mean Reward", row=2, col=1)
    fig.update_yaxes(title_text="Mean Length", row=2, col=2)
    
    fig.update_layout(
        title_text="Mountain Car PPO Training Dashboard",
        height=800,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

# ============================================================================
# Hyperparameters Visualization
# ============================================================================

def visualize_hyperparameters(model_path="./models/mountain_car_ppo_final.zip"):
    """
    Create an interactive visualization of model hyperparameters.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    model = PPO.load(model_path)
    
    # Extract hyperparameters
    hyperparams_raw = {
        'Learning Rate': model.learning_rate,
        'n_steps': model.n_steps,
        'Batch Size': model.batch_size,
        'n_epochs': model.n_epochs,
        'Gamma (Discount)': model.gamma,
        'GAE Lambda': model.gae_lambda,
        'Clip Range': model.clip_range,
        'Entropy Coef': model.ent_coef,
        'Value Function Coef': model.vf_coef,
        'Max Grad Norm': model.max_grad_norm
    }
    
    # Convert schedule objects to float values
    def get_value(v):
        """Convert schedule objects or callables to float."""
        if hasattr(v, '__call__'):
            # It's a callable (schedule), call it to get current value
            try:
                return float(v(1.0))  # Pass a dummy progress value
            except:
                # If calling fails, try to get value attribute
                if hasattr(v, 'value'):
                    return float(v.value)
                return float(v)
        return float(v)
    
    # Convert all values to floats
    hyperparams = {k: get_value(v) for k, v in hyperparams_raw.items()}
    
    # Create bar chart
    fig = go.Figure()
    
    # Normalize values for better visualization
    values = list(hyperparams.values())
    labels = list(hyperparams.keys())
    
    # Create color scale based on value ranges
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Format text labels
    text_labels = []
    for v in values:
        if v < 0.001:
            text_labels.append(f'{v:.2e}')
        elif v < 1:
            text_labels.append(f'{v:.6f}')
        else:
            text_labels.append(f'{v:.2f}')
    
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=text_labels,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='PPO Hyperparameters',
        xaxis_title='Hyperparameter',
        yaxis_title='Value',
        height=600,
        xaxis_tickangle=-45,
        yaxis_type='log' if max(values) > 100 else 'linear'
    )
    
    return fig, hyperparams

# ============================================================================
# Agent Performance Visualization
# ============================================================================

def visualize_agent_performance(model_path="./models/mountain_car_ppo_final.zip", n_episodes=5):
    """
    Visualize agent performance with interactive plots.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    model = PPO.load(model_path)
    env = gym.make("MountainCar-v0")
    
    all_episodes_data = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        positions = []
        velocities = []
        actions = []
        rewards = []
        timesteps = []
        
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            positions.append(obs[0])
            velocities.append(obs[1])
            actions.append(action)
            timesteps.append(step)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            step += 1
        
        all_episodes_data.append({
            'episode': episode + 1,
            'positions': positions,
            'velocities': velocities,
            'actions': actions,
            'rewards': rewards,
            'timesteps': timesteps,
            'total_reward': sum(rewards),
            'length': len(rewards)
        })
    
    env.close()
    
    # Create interactive visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Position Over Time', 'Velocity Over Time',
                       'Phase Space (Position vs Velocity)', 'Actions Over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot first episode in detail
    ep = all_episodes_data[0]
    
    # Position over time
    fig.add_trace(
        go.Scatter(
            x=ep['timesteps'],
            y=ep['positions'],
            mode='lines+markers',
            name='Position',
            line=dict(color='blue', width=2),
            hovertemplate='Step: %{x}<br>Position: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Goal", row=1, col=1)
    
    # Velocity over time
    fig.add_trace(
        go.Scatter(
            x=ep['timesteps'],
            y=ep['velocities'],
            mode='lines+markers',
            name='Velocity',
            line=dict(color='green', width=2),
            hovertemplate='Step: %{x}<br>Velocity: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Phase space for all episodes
    action_names = ['Left', 'No Accel', 'Right']
    colors_phase = ['blue', 'orange', 'green', 'red', 'purple']
    
    for i, ep in enumerate(all_episodes_data):
        fig.add_trace(
            go.Scatter(
                x=ep['positions'],
                y=ep['velocities'],
                mode='lines+markers',
                name=f'Episode {ep["episode"]}',
                line=dict(color=colors_phase[i % len(colors_phase)], width=2),
                marker=dict(size=4),
                hovertemplate='Position: %{x:.3f}<br>Velocity: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Goal", row=2, col=1)
    
    # Actions over time
    action_colors = ['red', 'gray', 'blue']
    for action_val in [0, 1, 2]:
        action_times = [i for i, a in enumerate(ep['actions']) if a == action_val]
        if action_times:
            fig.add_trace(
                go.Scatter(
                    x=[ep['timesteps'][i] for i in action_times],
                    y=[action_val] * len(action_times),
                    mode='markers',
                    name=action_names[action_val],
                    marker=dict(size=8, color=action_colors[action_val], symbol='square'),
                    hovertemplate='Step: %{x}<br>Action: %{text}<extra></extra>',
                    text=[action_names[action_val]] * len(action_times)
                ),
                row=2, col=2
            )
    
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=2)
    fig.update_xaxes(title_text="Position", row=2, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=2)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Velocity", row=1, col=2)
    fig.update_yaxes(title_text="Velocity", row=2, col=1)
    fig.update_yaxes(title_text="Action", row=2, col=2, tickmode='array', 
                     tickvals=[0, 1, 2], ticktext=action_names)
    
    fig.update_layout(
        title_text=f'Agent Performance - {n_episodes} Episodes',
        height=900,
        showlegend=True
    )
    
    return fig, all_episodes_data

# ============================================================================
# Summary Statistics Dashboard
# ============================================================================

def create_summary_dashboard(model_path="./models/mountain_car_ppo_final.zip", 
                             metrics_file="./training_logs/training_metrics.json"):
    """
    Create a comprehensive summary dashboard.
    """
    # Load training metrics if available
    training_data = None
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            training_data = json.load(f)
    
    # Get agent performance
    performance_data = None
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        env = gym.make("MountainCar-v0")
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(20):
            obs, info = env.reset()
            done = False
            total_reward = 0
            length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                length += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
        
        env.close()
        performance_data = {
            'rewards': episode_rewards,
            'lengths': episode_lengths
        }
    
    # Create summary dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Progress', 'Evaluation Results',
                       'Reward Distribution', 'Episode Length Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Training progress
    if training_data and training_data['mean_rewards']:
        fig.add_trace(
            go.Scatter(
                x=training_data['timesteps'][:len(training_data['mean_rewards'])],
                y=training_data['mean_rewards'],
                mode='lines',
                name='Mean Reward',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Evaluation results
    if performance_data:
        fig.add_trace(
            go.Box(
                y=performance_data['rewards'],
                name='Episode Rewards',
                boxmean='sd'
            ),
            row=1, col=2
        )
        
        # Reward distribution
        fig.add_trace(
            go.Histogram(
                x=performance_data['rewards'],
                name='Reward Distribution',
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # Length distribution
        fig.add_trace(
            go.Histogram(
                x=performance_data['lengths'],
                name='Length Distribution',
                nbinsx=20
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="Training Summary Dashboard",
        height=800,
        showlegend=True
    )
    
    return fig, performance_data

# ============================================================================
# Main Interactive Dashboard
# ============================================================================

def launch_interactive_dashboard():
    """
    Launch an interactive HTML dashboard with all visualizations.
    """
    print("Creating interactive dashboard...")
    
    # Create all visualizations
    visualizations = {}
    
    # Training dashboard
    if os.path.exists("./training_logs/training_metrics.json"):
        print("Loading training metrics...")
        visualizations['training'] = create_training_dashboard()
    
    # Hyperparameters
    if os.path.exists("./models/mountain_car_ppo_final.zip"):
        print("Loading model hyperparameters...")
        visualizations['hyperparams'] = visualize_hyperparameters()
        
        print("Evaluating agent performance...")
        visualizations['performance'] = visualize_agent_performance()
        
        print("Creating summary dashboard...")
        visualizations['summary'] = create_summary_dashboard()
    
    # Save all visualizations to HTML
    output_dir = "./visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    if 'training' in visualizations and visualizations['training']:
        visualizations['training'].write_html(f"{output_dir}/training_dashboard.html")
        print(f"[OK] Saved: {output_dir}/training_dashboard.html")
    
    if 'hyperparams' in visualizations and visualizations['hyperparams']:
        fig, params = visualizations['hyperparams']
        fig.write_html(f"{output_dir}/hyperparameters.html")
        print(f"[OK] Saved: {output_dir}/hyperparameters.html")
        print("\nHyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    if 'performance' in visualizations and visualizations['performance']:
        fig, data = visualizations['performance']
        fig.write_html(f"{output_dir}/agent_performance.html")
        print(f"[OK] Saved: {output_dir}/agent_performance.html")
    
    if 'summary' in visualizations and visualizations['summary']:
        fig, data = visualizations['summary']
        fig.write_html(f"{output_dir}/summary_dashboard.html")
        print(f"[OK] Saved: {output_dir}/summary_dashboard.html")
        
        if data:
            print(f"\nPerformance Statistics:")
            print(f"  Mean Reward: {np.mean(data['rewards']):.2f} ± {np.std(data['rewards']):.2f}")
            print(f"  Mean Length: {np.mean(data['lengths']):.2f} ± {np.std(data['lengths']):.2f}")
            print(f"  Success Rate: {sum(1 for r in data['rewards'] if r > -200) / len(data['rewards']) * 100:.1f}%")
    
    # Create main dashboard HTML
    create_main_dashboard_html(output_dir)
    
    print(f"\n{'='*60}")
    print("Interactive dashboards created!")
    print(f"{'='*60}")
    print(f"\nOpen {output_dir}/main_dashboard.html in your browser to view all visualizations!")

def create_main_dashboard_html(output_dir):
    """Create a main HTML file that links to all dashboards."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Mountain Car PPO - Interactive Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .dashboard-card {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.3s;
        }
        .dashboard-card:hover {
            transform: scale(1.05);
        }
        .dashboard-card a {
            color: white;
            text-decoration: none;
            font-size: 18px;
            font-weight: bold;
        }
        .dashboard-card a:hover {
            text-decoration: underline;
        }
        .icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 Mountain Car PPO - Interactive Dashboard</h1>
        <div class="dashboard-grid">
            <div class="dashboard-card">
                <div class="icon">📊</div>
                <a href="training_dashboard.html">Training Progress</a>
                <p>View training metrics and learning curves</p>
            </div>
            <div class="dashboard-card">
                <div class="icon">⚙️</div>
                <a href="hyperparameters.html">Hyperparameters</a>
                <p>Explore model configuration</p>
            </div>
            <div class="dashboard-card">
                <div class="icon">🎯</div>
                <a href="agent_performance.html">Agent Performance</a>
                <p>See how the agent performs</p>
            </div>
            <div class="dashboard-card">
                <div class="icon">📈</div>
                <a href="summary_dashboard.html">Summary Dashboard</a>
                <p>Comprehensive overview</p>
            </div>
        </div>
    </div>
</body>
</html>
    """
    
    with open(f"{output_dir}/main_dashboard.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    launch_interactive_dashboard()

