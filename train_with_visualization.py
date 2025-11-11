"""
Train PPO Agent with Real-time Visualization
Shows the car learning to climb the hill in real-time
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import numpy as np
import os
import json

# Import training monitor from main script
import sys
sys.path.append(os.path.dirname(__file__))
from mountain_car_ppo import TrainingMonitor

class VisualTrainingCallback(BaseCallback):
    """
    Callback to visualize training progress in real-time.
    Shows the agent playing every N episodes.
    """
    def __init__(self, eval_env, eval_freq=50, verbose=0):
        super(VisualTrainingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check for completed episodes
        infos = self.locals.get('infos', [])
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                self.episode_count += 1
                
                # Show visualization every eval_freq episodes
                if self.episode_count % self.eval_freq == 0:
                    self.show_agent_progress()
        
        return True
    
    def show_agent_progress(self):
        """Show the agent playing one episode."""
        print(f"\n{'='*60}")
        print(f"Episode {self.episode_count} - Showing Agent Progress")
        print(f"{'='*60}")
        
        obs, info = self.eval_env.reset()
        done = False
        total_reward = 0
        step = 0
        max_position = obs[0]
        
        # Create environment with rendering for visualization
        render_env = gym.make("MountainCar-v0", render_mode="human")
        render_obs, _ = render_env.reset()
        
        # Use current policy to play
        while not done and step < 200:
            # Get action from current policy (we'll need to pass model)
            # For now, just show the environment state
            action = self.eval_env.action_space.sample()  # Random for demo
            
            render_obs, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            total_reward += reward
            max_position = max(max_position, render_obs[0])
            step += 1
            
            # Small delay for visualization
            import time
            time.sleep(0.05)
        
        render_env.close()
        
        success = "[SUCCESS]" if max_position >= 0.5 else "[FAILED]"
        print(f"Steps: {step}, Max Position: {max_position:.4f}, Reward: {total_reward:.2f} {success}")
        print(f"{'='*60}\n")

def train_with_visualization(show_progress=True, progress_freq=50):
    """
    Train PPO agent with optional real-time visualization.
    
    Args:
        show_progress: Whether to show agent playing during training
        progress_freq: How often to show progress (every N episodes)
    """
    print("=" * 60)
    print("Mountain Car - PPO Training with Visualization")
    print("=" * 60)
    
    # Create vectorized environment for training
    env = make_vec_env("MountainCar-v0", n_envs=4, seed=42)
    
    # Create evaluation environment (for callbacks)
    eval_env = gym.make("MountainCar-v0")
    
    # Create visualization environment (for showing progress)
    if show_progress:
        visual_env = gym.make("MountainCar-v0", render_mode="human")
    else:
        visual_env = None
    
    # PPO Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        seed=42
    )
    
    # Callbacks
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
    
    training_monitor = TrainingMonitor(log_dir="./training_logs/", verbose=1)
    
    # Progress visualization callback
    callbacks = [eval_callback, checkpoint_callback, training_monitor]
    
    if show_progress:
        progress_callback = ProgressVisualizer(visual_env, progress_freq)
        progress_callback.set_model(model)  # Set model reference
        callbacks.append(progress_callback)
    
    # Train
    print("\nStarting training...")
    print("Note: Training windows will appear periodically to show agent progress")
    print(f"Progress will be shown every {progress_freq} episodes\n")
    
    model.learn(
        total_timesteps=500000,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save("./models/mountain_car_ppo_final")
    print("\nTraining completed! Model saved.")
    
    if visual_env:
        visual_env.close()
    eval_env.close()
    
    return model, eval_env

class ProgressVisualizer(BaseCallback):
    """Shows agent playing during training."""
    def __init__(self, visual_env, eval_freq=50, verbose=0):
        super(ProgressVisualizer, self).__init__(verbose)
        self.visual_env = visual_env
        self.eval_freq = eval_freq
        self.episode_count = 0
        self.model = None  # Will be set by parent
        
    def set_model(self, model):
        """Set the model to use for predictions."""
        self.model = model
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if isinstance(info, dict) and 'episode' in info:
                self.episode_count += 1
                
                if self.episode_count % self.eval_freq == 0:
                    self.show_progress()
        
        return True
    
    def show_progress(self):
        """Show agent playing one episode using current model."""
        if self.model is None:
            return  # Can't show without model
        
        print(f"\n{'='*60}")
        print(f"Episode {self.episode_count} - Visualizing Agent Progress")
        print(f"{'='*60}")
        
        obs, _ = self.visual_env.reset()
        done = False
        total_reward = 0
        step = 0
        max_position = obs[0]
        
        import time
        
        while not done and step < 200:
            try:
                action, _ = self.model.predict(obs, deterministic=False)
            except:
                action = self.visual_env.action_space.sample()
            
            obs, reward, terminated, truncated, _ = self.visual_env.step(action)
            done = terminated or truncated
            total_reward += reward
            max_position = max(max_position, obs[0])
            step += 1
            time.sleep(0.03)  # Small delay for visualization
        
        success = "[SUCCESS]" if max_position >= 0.5 else "[FAILED]"
        print(f"Result: {success}")
        print(f"Steps: {step}, Max Position: {max_position:.4f}, Reward: {total_reward:.2f}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO with visualization')
    parser.add_argument('--no-visual', action='store_true', 
                       help='Disable real-time visualization')
    parser.add_argument('--freq', type=int, default=50,
                       help='Show progress every N episodes (default: 50)')
    
    args = parser.parse_args()
    
    train_with_visualization(
        show_progress=not args.no_visual,
        progress_freq=args.freq
    )

