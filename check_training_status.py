"""
Check Training Status
Diagnostic script to check if training has run and if metrics are available
"""

import os
import json
from pathlib import Path

def check_training_status():
    """Check the status of training and available metrics."""
    print("=" * 60)
    print("Training Status Check")
    print("=" * 60)
    
    # Check for training metrics
    metrics_file = "./training_logs/training_metrics.json"
    metrics_exists = os.path.exists(metrics_file)
    
    print(f"\n1. Training Metrics File:")
    print(f"   Path: {metrics_file}")
    print(f"   Exists: {'YES' if metrics_exists else 'NO'}")
    
    if metrics_exists:
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            num_episodes = len(metrics.get('episode_rewards', []))
            num_timesteps = metrics.get('timesteps', [])
            last_timestep = num_timesteps[-1] if num_timesteps else 0
            
            print(f"   Episodes recorded: {num_episodes}")
            print(f"   Last timestep: {last_timestep:,}")
            
            if num_episodes > 0:
                recent_rewards = metrics['episode_rewards'][-10:]
                mean_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"   Mean reward (last 10): {mean_reward:.2f}")
                print(f"   Best reward: {max(metrics['episode_rewards']):.2f}")
        except Exception as e:
            print(f"   Error reading file: {e}")
    else:
        print("   [WARNING] No training metrics found!")
        print("   This means training hasn't started or hasn't saved metrics yet.")
    
    # Check for model files
    print(f"\n2. Model Files:")
    model_dir = "./models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
        print(f"   Directory: {model_dir}")
        print(f"   Models found: {len(model_files)}")
        for model_file in model_files[:5]:  # Show first 5
            file_path = os.path.join(model_dir, model_file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"     - {model_file} ({size:.2f} MB)")
        if len(model_files) > 5:
            print(f"     ... and {len(model_files) - 5} more")
    else:
        print(f"   Directory not found: {model_dir}")
    
    # Check for TensorBoard logs
    print(f"\n3. TensorBoard Logs:")
    tb_dir = "./tensorboard_logs"
    if os.path.exists(tb_dir):
        tb_files = list(Path(tb_dir).rglob('*'))
        print(f"   Directory: {tb_dir}")
        print(f"   Log files found: {len(tb_files)}")
        if tb_files:
            print(f"   [OK] TensorBoard logs available")
            print(f"   View with: tensorboard --logdir {tb_dir}")
    else:
        print(f"   Directory not found: {tb_dir}")
    
    # Recommendations
    print(f"\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    if not metrics_exists:
        print("\n[INFO] No training metrics found. To get training progress:")
        print("   1. Run training: python mountain_car_ppo.py")
        print("   2. Wait for at least 5 episodes to complete")
        print("   3. Metrics will be saved automatically to training_logs/")
        print("   4. Run visualization: python visualize_training.py")
    else:
        print("\n[OK] Training metrics found!")
        print("   You can now:")
        print("   1. Run: python visualize_training.py")
        print("   2. Open: ./visualizations/main_dashboard.html")
        print("   3. View TensorBoard: tensorboard --logdir ./tensorboard_logs/")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_training_status()

