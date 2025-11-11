"""
Real-time Agent Visualization
Watch the trained agent play Mountain Car in real-time
"""

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import time
import os

def view_agent(model_path="./models/mountain_car_ppo_final.zip", 
               n_episodes=5, 
               render_mode="human",
               speed=0.05):
    """
    Visualize the trained agent playing Mountain Car.
    
    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to visualize
        render_mode: "human" for window, "rgb_array" for frames
        speed: Delay between steps (seconds)
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first by running: python mountain_car_ppo.py")
        return
    
    print("=" * 60)
    print("Mountain Car Agent Visualization")
    print("=" * 60)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    
    # Create environment
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    
    episode_stats = []
    
    for episode in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_position = obs[0]
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step_count += 1
            max_position = max(max_position, obs[0])
            
            # Display current state
            if step_count % 10 == 0:
                action_names = ['Left', 'No Accel', 'Right']
                print(f"Step {step_count:3d} | Position: {obs[0]:7.4f} | "
                      f"Velocity: {obs[1]:7.4f} | Action: {action_names[action]} | "
                      f"Reward: {total_reward:6.1f}")
            
            # Control playback speed
            if render_mode == "human":
                time.sleep(speed)
        
        # Episode summary
        success = "[SUCCESS]" if obs[0] >= 0.5 else "[FAILED]"
        print(f"\nEpisode {episode + 1} Complete:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Final Position: {obs[0]:.4f}")
        print(f"  Max Position: {max_position:.4f}")
        print(f"  Result: {success}")
        
        episode_stats.append({
            'episode': episode + 1,
            'reward': total_reward,
            'steps': step_count,
            'final_position': obs[0],
            'max_position': max_position,
            'success': obs[0] >= 0.5
        })
    
    env.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    success_rate = sum(1 for s in episode_stats if s['success']) / len(episode_stats) * 100
    mean_reward = np.mean([s['reward'] for s in episode_stats])
    mean_steps = np.mean([s['steps'] for s in episode_stats])
    
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean Reward: {mean_reward:.2f} ± {np.std([s['reward'] for s in episode_stats]):.2f}")
    print(f"Mean Steps: {mean_steps:.1f} ± {np.std([s['steps'] for s in episode_stats]):.1f}")
    print(f"Best Episode: Episode {max(episode_stats, key=lambda x: x['reward'])['episode']} "
          f"(Reward: {max(episode_stats, key=lambda x: x['reward'])['reward']:.2f})")

if __name__ == "__main__":
    import os
    import sys
    
    # Default settings
    model_path = "./models/mountain_car_ppo_final.zip"
    n_episodes = 5
    speed = 0.05
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        n_episodes = int(sys.argv[2])
    if len(sys.argv) > 3:
        speed = float(sys.argv[3])
    
    view_agent(model_path, n_episodes, render_mode="human", speed=speed)

