"""
Mountain Car Environment with PPO Algorithm
Using Gymnasium and Stable Baselines3
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import json

# Create output directory
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================================
# Environment Setup
# ============================================================================

def create_environment():
    """
    Create and configure the Mountain Car environment.
    
    Observation Space: Box([-1.2, -0.07], [0.6, 0.07], (2,), float32)
    - Index 0: Position of car along x-axis (range: -1.2 to 0.6)
    - Index 1: Velocity of car (range: -0.07 to 0.07)
    
    Action Space: Discrete(3)
    - 0: Accelerate to the left
    - 1: Don't accelerate
    - 2: Accelerate to the right
    """
    env = gym.make("MountainCar-v0", render_mode=None)
    return env

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
        self.last_save_step = 0
        
    def _on_step(self) -> bool:
        # Get info from the environment
        infos = self.locals.get('infos', [])
        if infos:
            for info in infos:
                if isinstance(info, dict) and 'episode' in info:
                    episode_info = info['episode']
                    if 'r' in episode_info and 'l' in episode_info:
                        reward = episode_info['r']
                        length = episode_info['l']
                        
                        self.episode_rewards.append(reward)
                        self.episode_lengths.append(length)
                        self.timesteps.append(self.num_timesteps)
                        
                        # Calculate running means
                        if len(self.episode_rewards) >= 10:
                            self.mean_rewards.append(np.mean(self.episode_rewards[-10:]))
                            self.mean_lengths.append(np.mean(self.episode_lengths[-10:]))
                        else:
                            self.mean_rewards.append(np.mean(self.episode_rewards))
                            self.mean_lengths.append(np.mean(self.episode_lengths))
                        
                        # Save metrics every 5 episodes or every 10000 steps
                        if (len(self.episode_rewards) % 5 == 0) or (self.num_timesteps - self.last_save_step >= 10000):
                            self.save_metrics()
                            self.last_save_step = self.num_timesteps
        
        return True
    
    def _on_training_end(self) -> None:
        """Save metrics when training ends."""
        self.save_metrics()
        if self.verbose > 0:
            print(f"\nTraining metrics saved to {os.path.join(self.log_dir, 'training_metrics.json')}")
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics = {
            'timesteps': self.timesteps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_rewards': self.mean_rewards,
            'mean_lengths': self.mean_lengths
        }
        metrics_path = os.path.join(self.log_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

# ============================================================================
# Training Configuration
# ============================================================================

def train_ppo_agent():
    """
    Train a PPO agent on the Mountain Car environment.
    """
    print("=" * 60)
    print("Mountain Car - PPO Training")
    print("=" * 60)
    
    # Create vectorized environment for faster training
    # Using 4 parallel environments
    env = make_vec_env("MountainCar-v0", n_envs=4, seed=42)
    
    # Create evaluation environment
    eval_env = gym.make("MountainCar-v0")
    
    # PPO Hyperparameters
    model = PPO(
        "MlpPolicy",              # Multi-layer perceptron policy
        env,
        learning_rate=3e-4,       # Learning rate
        n_steps=2048,             # Steps per update
        batch_size=64,            # Batch size for training
        n_epochs=10,              # Number of epochs per update
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE lambda parameter
        clip_range=0.2,           # PPO clipping parameter
        ent_coef=0.01,            # Entropy coefficient
        vf_coef=0.5,              # Value function coefficient
        max_grad_norm=0.5,        # Gradient clipping
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        seed=42
    )
    
    # Callbacks for evaluation and checkpointing
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./results/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="mountain_car_ppo"
    )
    
    # Training monitor callback
    training_monitor = TrainingMonitor(log_dir="./training_logs/")
    
    # Train the model
    print("\nStarting training...")
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback, training_monitor],
        progress_bar=True
    )
    
    # Save the final model
    model.save("./models/mountain_car_ppo_final")
    print("\nTraining completed! Model saved.")
    
    return model, eval_env

# ============================================================================
# Evaluation and Testing
# ============================================================================

def evaluate_agent(model, env, n_episodes=10, render=False):
    """
    Evaluate the trained agent.
    
    Returns:
        - Mean reward
        - Mean episode length
        - List of episode rewards
        - List of episode data (states, actions, rewards)
    """
    episode_rewards = []
    episode_lengths = []
    episode_data = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_states = []
        episode_actions = []
        episode_rewards_list = []
        
        while not done:
            if render:
                env.render()
            
            # Store state
            episode_states.append(obs.copy())
            
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_rewards_list.append(reward)
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_data.append({
            'states': np.array(episode_states),
            'actions': np.array(episode_actions),
            'rewards': np.array(episode_rewards_list),
            'total_reward': episode_reward,
            'length': episode_length
        })
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nMean Reward: {mean_reward:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {mean_length:.2f} ± {np.std(episode_lengths):.2f}")
    
    return mean_reward, mean_length, episode_rewards, episode_data

# ============================================================================
# Visualization
# ============================================================================

def plot_training_results(episode_data):
    """
    Plot training results including states, actions, and rewards.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Position over time for first episode
    episode = episode_data[0]
    positions = episode['states'][:, 0]
    velocities = episode['states'][:, 1]
    
    axes[0, 0].plot(positions, label='Position')
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='Goal (0.5)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].set_title('Car Position Over Time (Episode 1)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Velocity over time
    axes[0, 1].plot(velocities, label='Velocity', color='orange')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Car Velocity Over Time (Episode 1)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Actions over time
    actions = episode['actions']
    action_names = ['Left', 'No Accel', 'Right']
    axes[1, 0].plot(actions, marker='o', markersize=3, linestyle='None')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Action')
    axes[1, 0].set_title('Actions Over Time (Episode 1)')
    axes[1, 0].set_yticks([0, 1, 2])
    axes[1, 0].set_yticklabels(action_names)
    axes[1, 0].grid(True)
    
    # Plot 4: Cumulative reward
    cumulative_reward = np.cumsum(episode['rewards'])
    axes[1, 1].plot(cumulative_reward)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Cumulative Reward Over Time (Episode 1)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("./results/training_analysis.png", dpi=300)
    print("\nVisualization saved to ./results/training_analysis.png")
    plt.close()

def plot_phase_space(episode_data):
    """
    Plot phase space diagram (position vs velocity).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, episode in enumerate(episode_data[:5]):  # Plot first 5 episodes
        positions = episode['states'][:, 0]
        velocities = episode['states'][:, 1]
        ax.plot(positions, velocities, alpha=0.6, label=f'Episode {i+1}')
    
    ax.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Goal Position')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Phase Space: Position vs Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./results/phase_space.png", dpi=300)
    print("Phase space plot saved to ./results/phase_space.png")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Train the agent
    model, eval_env = train_ppo_agent()
    
    # Evaluate the trained agent
    print("\n" + "=" * 60)
    print("Evaluating Trained Agent")
    print("=" * 60)
    mean_reward, mean_length, episode_rewards, episode_data = evaluate_agent(
        model, eval_env, n_episodes=10, render=False
    )
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    plot_training_results(episode_data)
    plot_phase_space(episode_data)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Success Rate: {sum(1 for r in episode_rewards if r > -200) / len(episode_rewards) * 100:.1f}%")
    print(f"Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {min(episode_rewards):.2f}")
    
    eval_env.close()
    print("\nEvaluation complete!")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Interactive Visualizations")
    print("=" * 60)
    try:
        from visualize_training import launch_interactive_dashboard
        launch_interactive_dashboard()
    except ImportError:
        print("Note: Install plotly to generate interactive visualizations:")
        print("  pip install plotly pandas")
    except Exception as e:
        print(f"Could not generate visualizations: {e}")

