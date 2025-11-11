"""
Extract Training Metrics from TensorBoard Logs
This script reads TensorBoard event files and creates training_metrics.json
"""

import os
import json
from pathlib import Path
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

def extract_metrics_from_tensorboard(log_dir="./tensorboard_logs"):
    """
    Extract training metrics from TensorBoard logs.
    """
    if not TENSORBOARD_AVAILABLE:
        print("Cannot extract metrics: TensorBoard not installed")
        return None
    
    if not os.path.exists(log_dir):
        print(f"TensorBoard log directory not found: {log_dir}")
        return None
    
    # Find the most recent run
    log_paths = list(Path(log_dir).glob("*/events.out.tfevents.*"))
    if not log_paths:
        print(f"No TensorBoard event files found in {log_dir}")
        return None
    
    # Use the most recent log file
    latest_log = max(log_paths, key=lambda p: p.stat().st_mtime)
    log_dir_path = latest_log.parent
    
    print(f"Reading TensorBoard logs from: {log_dir_path}")
    
    # Load event accumulator
    ea = EventAccumulator(str(log_dir_path))
    ea.Reload()
    
    # Extract scalar metrics
    scalars = ea.Tags()['scalars']
    
    metrics = {
        'timesteps': [],
        'episode_rewards': [],
        'episode_lengths': [],
        'mean_rewards': [],
        'mean_lengths': []
    }
    
    # Try to find episode reward data
    reward_scalars = [s for s in scalars if 'reward' in s.lower() or 'episode' in s.lower()]
    
    if reward_scalars:
        print(f"Found reward-related scalars: {reward_scalars}")
        
        # Try to get episode/reward data
        for scalar_name in reward_scalars:
            scalar_events = ea.Scalars(scalar_name)
            for event in scalar_events:
                step = int(event.step)
                value = float(event.value)
                
                # Try to identify what this metric is
                if 'reward' in scalar_name.lower() and 'mean' not in scalar_name.lower():
                    metrics['episode_rewards'].append(value)
                    metrics['timesteps'].append(step)
                elif 'length' in scalar_name.lower() or 'episode_length' in scalar_name.lower():
                    if len(metrics['episode_lengths']) < len(metrics['episode_rewards']):
                        metrics['episode_lengths'].append(value)
                elif 'mean' in scalar_name.lower() and 'reward' in scalar_name.lower():
                    metrics['mean_rewards'].append(value)
    
    # If we didn't find episode rewards, try rollouts
    if not metrics['episode_rewards']:
        rollout_scalars = [s for s in scalars if 'rollout' in s.lower()]
        for scalar_name in rollout_scalars:
            scalar_events = ea.Scalars(scalar_name)
            for event in scalar_events:
                step = int(event.step)
                value = float(event.value)
                
                if 'ep_rew_mean' in scalar_name or 'rollout/ep_rew_mean' in scalar_name:
                    metrics['mean_rewards'].append(value)
                    metrics['timesteps'].append(step)
                elif 'ep_len_mean' in scalar_name or 'rollout/ep_len_mean' in scalar_name:
                    metrics['mean_lengths'].append(value)
    
    # Calculate episode rewards from mean if we have that
    if metrics['mean_rewards'] and not metrics['episode_rewards']:
        # Estimate episode rewards from mean (approximation)
        for i, mean_reward in enumerate(metrics['mean_rewards']):
            # Approximate: assume mean is close to individual rewards
            metrics['episode_rewards'].append(mean_reward)
            if i < len(metrics['timesteps']):
                metrics['timesteps'].append(metrics['timesteps'][i])
    
    # Save metrics
    if metrics['episode_rewards'] or metrics['mean_rewards']:
        output_dir = "./training_logs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'training_metrics.json')
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n[OK] Extracted metrics saved to: {output_file}")
        print(f"   Episodes: {len(metrics['episode_rewards'])}")
        print(f"   Mean rewards: {len(metrics['mean_rewards'])}")
        return metrics
    else:
        print("\n[WARNING] Could not extract episode metrics from TensorBoard logs")
        print("   Available scalars:", scalars)
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Extracting Training Metrics from TensorBoard Logs")
    print("=" * 60)
    metrics = extract_metrics_from_tensorboard()
    
    if metrics:
        print("\n[SUCCESS] Metrics extracted! You can now run:")
        print("   python visualize_training.py")
    else:
        print("\n[INFO] Could not extract metrics. Options:")
        print("   1. Re-run training: python mountain_car_ppo.py")
        print("   2. Check TensorBoard manually: tensorboard --logdir ./tensorboard_logs/")

