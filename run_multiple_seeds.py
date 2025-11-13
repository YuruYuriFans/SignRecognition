"""
run_multiple_seeds.py - Train and evaluate models with multiple random seeds
Then perform paired t-test to determine statistical significance.

This script leverages your existing train.py, tune.py, and predict.py scripts.

Usage:
    python run_multiple_seeds.py --n_seeds 10
    python run_multiple_seeds.py --n_seeds 5 --skip_training  # Only evaluate
"""

import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
import re

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def run_command(cmd, description, env=None):
    """Run a shell command and capture output."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            env=env
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def train_basic_lenet(seed, args):
    """Train basic LeNet with a specific seed (auto-detects trained model filename)."""
    print(f"\n{'#'*80}")
    print(f"TRAINING BASIC LENET - SEED {seed}")
    print(f"{'#'*80}")

    # Set random seed via environment variable
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)

    # Run the training command
    cmd = [
        sys.executable, 'train.py',
        '--epochs', '3',
        '--batch_size', '128',
        '--lr', '0.001',
        # '--dropout', '0.5',
        # '--weight_decay', '0.0001',
        # '--patience', '20',
        # '--min_delta', '0.0',
        # '--optimizer', 'adam',
        '--aug', 'basic'
    ]

    success = run_command(cmd, f"Training Basic LeNet (seed {seed})", env=env)

    if not success:
        print(f"❌ Training failed for seed {seed}")
        return None

    # --- Auto-detect the basic model file ---
    if not os.path.exists('trained_models'):
        print("❌ No 'trained_models' directory found after training.")
        return None

    basic_candidates = [
        f for f in os.listdir('trained_models')
        if f.startswith('best_lenet_basic') and f.endswith('.pth')
    ]

    if not basic_candidates:
        print("❌ No basic models found in 'trained_models/'.")
        return None

    # Pick the most recently modified trained model
    latest_basic_path = max(
        (os.path.join('trained_models', f) for f in basic_candidates),
        key=os.path.getmtime
    )

    seed_basic_path = f'trained_models/best_lenet_basic_seed{seed}.pth'

    # Remove old file if it exists
    if os.path.exists(seed_basic_path):
        os.remove(seed_basic_path)

    try:
        shutil.copy2(latest_basic_path, seed_basic_path)
        print(f"✓ Copied {latest_basic_path} -> {seed_basic_path}")
        return seed_basic_path
    except Exception as e:
        print(f"❌ Error copying basic model: {e}")
        import traceback
        traceback.print_exc()
        return None


def tune_lenet(seed, basic_model_path, args):
    """Fine-tune LeNet with a specific seed (auto-detects tuned model filename)."""
    print(f"\n{'#'*80}")
    print(f"FINE-TUNING LENET - SEED {seed}")
    print(f"{'#'*80}")

    if not os.path.exists(basic_model_path):
        print(f"❌ Error: Basic model not found at {basic_model_path}")
        return None

    # Set random seed via environment variable
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)

    # Use tune.py with your best parameters
    cmd = [
        sys.executable, 'tune.py',
        '--ckpt', basic_model_path,
        '--epochs', '3',
        '--batch_size', '128',
        '--lr', '0.0003',
        '--dropout', '0.6',
        '--weight_decay', '0.0003',
        '--patience', '20',
        '--min_delta', '0.0',
        '--optimizer', 'adamw',
        '--aug', 'advanced'
    ]

    success = run_command(cmd, f"Fine-tuning LeNet (seed {seed})", env=env)

    if not success:
        print(f"❌ Fine-tuning failed for seed {seed}")
        return None

    # --- Auto-detect the tuned model file ---
    if not os.path.exists('trained_models'):
        print("❌ No 'trained_models' directory found after tuning.")
        return None

    tuned_candidates = [
        f for f in os.listdir('trained_models')
        if f.startswith('best_lenet_tuned') and f.endswith('.pth')
    ]

    if not tuned_candidates:
        print("❌ No tuned models found in 'trained_models/'.")
        return None

    # Pick the most recently modified tuned model
    latest_tuned_path = max(
        (os.path.join('trained_models', f) for f in tuned_candidates),
        key=os.path.getmtime
    )

    seed_tuned_path = f'trained_models/best_lenet_tuned_seed{seed}.pth'

    # Remove old file if it exists
    if os.path.exists(seed_tuned_path):
        os.remove(seed_tuned_path)

    try:
        shutil.copy2(latest_tuned_path, seed_tuned_path)
        print(f"✓ Copied {latest_tuned_path} -> {seed_tuned_path}")
        return seed_tuned_path
    except Exception as e:
        print(f"❌ Error copying tuned model: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_models(seed, basic_model_path, tuned_model_path, args):
    """Evaluate both models using predict.py and extract accuracies."""
    print(f"\n{'#'*80}")
    print(f"EVALUATING MODELS - SEED {seed}")
    print(f"{'#'*80}")
    
    # Verify models exist before evaluation
    if not os.path.exists(basic_model_path):
        print(f"❌ ERROR: Basic model not found: {basic_model_path}")
        return {
            'seed': seed,
            'basic_accuracy': None,
            'tuned_accuracy': None,
            'basic_model_path': basic_model_path,
            'tuned_model_path': tuned_model_path
        }
    
    if not os.path.exists(tuned_model_path):
        print(f"❌ ERROR: Tuned model not found: {tuned_model_path}")
        return {
            'seed': seed,
            'basic_accuracy': None,
            'tuned_accuracy': None,
            'basic_model_path': basic_model_path,
            'tuned_model_path': tuned_model_path
        }
    
    print(f"✓ Basic model found: {basic_model_path}")
    print(f"✓ Tuned model found: {tuned_model_path}")
    
    results = {
        'seed': seed,
        'basic_accuracy': None,
        'tuned_accuracy': None,
        'basic_model_path': basic_model_path,
        'tuned_model_path': tuned_model_path
    }
    
    # Create a temporary directory with only these two models
    temp_dir = f'temp_eval_seed{seed}'
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Copy models to temp directory
        temp_basic = os.path.join(temp_dir, f'basic_seed{seed}.pth')
        temp_tuned = os.path.join(temp_dir, f'tuned_seed{seed}.pth')
        
        shutil.copy2(basic_model_path, temp_basic)
        shutil.copy2(tuned_model_path, temp_tuned)
        
        # Temporarily rename trained_models to avoid predict.py scanning all models
        trained_models_backup = None
        if os.path.exists('trained_models'):
            trained_models_backup = 'trained_models_backup_temp'
            if os.path.exists(trained_models_backup):
                shutil.rmtree(trained_models_backup)
            shutil.move('trained_models', trained_models_backup)
        
        # Create new trained_models with only our two models
        os.makedirs('trained_models', exist_ok=True)
        shutil.copy2(basic_model_path, os.path.join('trained_models', f'model_basic_seed{seed}.pth'))
        shutil.copy2(tuned_model_path, os.path.join('trained_models', f'model_tuned_seed{seed}.pth'))
        
        # Run predict.py
        cmd = [sys.executable, 'predict.py']
        
        # Capture output to parse accuracies
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        output = result.stdout
        print(output)
        
        # Parse accuracies from predict.py output
        # Look for lines like "TestAcc" in the summary table
        # Format: model_name ... TestAcc ... (values)
        
        lines = output.split('\n')
        for line in lines:
            if 'model_basic_seed' in line.lower() and seed in str(line):
                # Extract test accuracy
                match = re.search(r'(\d+\.\d{4})', line)
                if match:
                    results['basic_accuracy'] = float(match.group(1))
            elif 'model_tuned_seed' in line.lower() and seed in str(line):
                match = re.search(r'(\d+\.\d{4})', line)
                if match:
                    results['tuned_accuracy'] = float(match.group(1))
        
        # Alternative: parse from records/predicting_results.txt
        if results['basic_accuracy'] is None or results['tuned_accuracy'] is None:
            pred_results_path = 'records/predicting_results.txt'
            if os.path.exists(pred_results_path):
                with open(pred_results_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for line in lines:
                        if f'basic_seed{seed}' in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                try:
                                    val = float(part)
                                    if 0 < val < 1:
                                        results['basic_accuracy'] = val
                                        break
                                except ValueError:
                                    continue
                        elif f'tuned_seed{seed}' in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                try:
                                    val = float(part)
                                    if 0 < val < 1:
                                        results['tuned_accuracy'] = val
                                        break
                                except ValueError:
                                    continue
        
        print(f"\n{'='*80}")
        print(f"SEED {seed} EVALUATION RESULTS:")
        print(f"  Basic Model: {results['basic_accuracy']}")
        print(f"  Tuned Model: {results['tuned_accuracy']}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore trained_models directory
        if os.path.exists('trained_models'):
            shutil.rmtree('trained_models')
        if trained_models_backup and os.path.exists(trained_models_backup):
            shutil.move(trained_models_backup, 'trained_models')
        
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple seed experiments and statistical test'
    )
    parser.add_argument('--n_seeds', type=int, default=10, 
                       help='Number of random seeds to test')
    parser.add_argument('--start_seed', type=int, default=42, 
                       help='Starting seed value')
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip training, only evaluate existing models')
    parser.add_argument('--results_dir', type=str, default='seed_results', 
                       help='Directory to store results')
    
    # Training parameters
    parser.add_argument('--basic_epochs', type=int, default=30, 
                       help='Epochs for basic training')
    parser.add_argument('--basic_batch_size', type=int, default=128, 
                       help='Batch size for basic training')
    parser.add_argument('--basic_lr', type=float, default=0.001, 
                       help='Learning rate for basic training')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MULTI-SEED TRAINING AND EVALUATION")
    print(f"{'='*80}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Starting seed: {args.start_seed}")
    print(f"Skip training: {args.skip_training}")
    print(f"Results directory: {args.results_dir}")
    print(f"{'='*80}\n")
    
    # Storage for results
    all_results = []
    basic_accuracies = []
    tuned_accuracies = []
    
    # Run experiments for each seed
    for i in range(args.n_seeds):
        seed = args.start_seed + i
        
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT {i+1}/{args.n_seeds} - SEED {seed}")
        print(f"{'#'*80}\n")
        
        basic_model_path = f'trained_models/best_lenet_basic_seed{seed}.pth'
        tuned_model_path = f'trained_models/best_lenet_tuned_seed{seed}.pth'
        
        # Train or check for existing models
        if not args.skip_training:
            # Train basic model
            print(f"\n[Step 1/3] Training basic model...")
            basic_path = train_basic_lenet(seed, args)
            if not basic_path:
                print(f"❌ Failed to train basic model for seed {seed}, skipping...")
                continue
            basic_model_path = basic_path
            
            # Verify basic model exists
            if not os.path.exists(basic_model_path):
                print(f"❌ Basic model not found after training: {basic_model_path}")
                print(f"Checking trained_models directory...")
                if os.path.exists('trained_models'):
                    files = os.listdir('trained_models')
                    print(f"Available files: {files}")
                continue
            
            # Fine-tune model
            print(f"\n[Step 2/3] Fine-tuning model...")
            tuned_path = tune_lenet(seed, basic_model_path, args)
            if not tuned_path:
                print(f"❌ Failed to tune model for seed {seed}, skipping...")
                continue
            tuned_model_path = tuned_path
            
            # Verify tuned model exists
            if not os.path.exists(tuned_model_path):
                print(f"❌ Tuned model not found after training: {tuned_model_path}")
                print(f"Checking trained_models directory...")
                if os.path.exists('trained_models'):
                    files = os.listdir('trained_models')
                    print(f"Available files: {files}")
                continue
        else:
            # Check if models exist
            if not os.path.exists(basic_model_path):
                print(f"❌ Basic model not found: {basic_model_path}, skipping seed {seed}")
                print(f"Checking trained_models directory...")
                if os.path.exists('trained_models'):
                    files = os.listdir('trained_models')
                    print(f"Available files: {files}")
                continue
            if not os.path.exists(tuned_model_path):
                print(f"❌ Tuned model not found: {tuned_model_path}, skipping seed {seed}")
                print(f"Checking trained_models directory...")
                if os.path.exists('trained_models'):
                    files = os.listdir('trained_models')
                    print(f"Available files: {files}")
                continue
        
        # Evaluate both models
        print(f"\n[Step 3/3] Evaluating models...")
        eval_results = evaluate_models(seed, basic_model_path, tuned_model_path, args)
        
        if eval_results['basic_accuracy'] is not None and eval_results['tuned_accuracy'] is not None:
            basic_acc = eval_results['basic_accuracy']
            tuned_acc = eval_results['tuned_accuracy']
            improvement = tuned_acc - basic_acc
            
            result_entry = {
                'seed': seed,
                'basic_accuracy': basic_acc,
                'tuned_accuracy': tuned_acc,
                'improvement': improvement,
                'improvement_pct': improvement * 100
            }
            
            all_results.append(result_entry)
            basic_accuracies.append(basic_acc)
            tuned_accuracies.append(tuned_acc)
            
            print(f"\n{'='*80}")
            print(f"SEED {seed} SUMMARY:")
            print(f"  Basic: {basic_acc:.4f} ({basic_acc*100:.2f}%)")
            print(f"  Tuned: {tuned_acc:.4f} ({tuned_acc*100:.2f}%)")
            print(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
            print(f"{'='*80}")
        else:
            print(f"Warning: Could not extract accuracies for seed {seed}")
    
    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    
    if len(basic_accuracies) < 2:
        print("\n" + "="*80)
        print("ERROR: Need at least 2 successful runs for statistical testing")
        print(f"Only {len(basic_accuracies)} paired results collected")
        print("="*80)
        return
    
    print(f"\n{'='*80}")
    print(f"COLLECTED {len(basic_accuracies)} PAIRED RESULTS")
    print(f"{'='*80}")
    
    # Save results to CSV
    results_csv = os.path.join(args.results_dir, 'seed_results.csv')
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to: {results_csv}")
    
    # Print summary table
    print(f"\n{'Seed':<8} {'Basic Acc':<25} {'Tuned Acc':<25} {'Improvement':<20}")
    print("-" * 80)
    for entry in all_results:
        print(f"{entry['seed']:<8} "
              f"{entry['basic_accuracy']:.4f} ({entry['basic_accuracy']*100:6.2f}%)  "
              f"{entry['tuned_accuracy']:.4f} ({entry['tuned_accuracy']*100:6.2f}%)  "
              f"{entry['improvement']:+.4f} ({entry['improvement_pct']:+6.2f}%)")
    
    # Perform statistical test
    print(f"\n{'='*80}")
    print("PERFORMING PAIRED T-TEST")
    print(f"{'='*80}")
    
    try:
        from statistical_test import paired_ttest, print_results, plot_results
        
        results = paired_ttest(basic_accuracies, tuned_accuracies)
        print_results(results)
        
        # Generate plots
        plot_results(basic_accuracies, tuned_accuracies)
        plot_path = os.path.join(args.results_dir, 'statistical_test_results.png')
        if os.path.exists('statistical_test_results.png'):
            shutil.move('statistical_test_results.png', plot_path)
            print(f"\n✓ Plots saved to: {plot_path}")
        
        # Save statistical results to JSON
        stats_json = os.path.join(args.results_dir, 'statistical_results.json')
        with open(stats_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Statistical results saved to: {stats_json}")
        
    except ImportError:
        print("\nWarning: statistical_test.py not found. Performing basic analysis...")
        
        from scipy import stats
        
        basic_mean = np.mean(basic_accuracies)
        tuned_mean = np.mean(tuned_accuracies)
        basic_std = np.std(basic_accuracies, ddof=1)
        tuned_std = np.std(tuned_accuracies, ddof=1)
        
        differences = np.array(tuned_accuracies) - np.array(basic_accuracies)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(tuned_accuracies, basic_accuracies)
        
        print(f"\nBasic LeNet: {basic_mean:.4f} ± {basic_std:.4f}")
        print(f"Tuned LeNet: {tuned_mean:.4f} ± {tuned_std:.4f}")
        print(f"Mean Improvement: {mean_diff:.4f} ± {std_diff:.4f}")
        print(f"\nt-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"\n✓ SIGNIFICANT: Improvement is statistically significant (p < 0.05)")
        else:
            print(f"\n✗ NOT SIGNIFICANT: Improvement is not statistically significant (p >= 0.05)")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved in: {args.results_dir}/")
    print(f"  - seed_results.csv: Raw data")
    print(f"  - statistical_results.json: Statistical analysis")
    print(f"  - statistical_test_results.png: Visualization")
    print(f"\nAll trained models saved in: trained_models/")
    print(f"  - best_lenet_basic_seed*.pth")
    print(f"  - best_lenet_tuned_seed*.pth")


if __name__ == '__main__':
    main()