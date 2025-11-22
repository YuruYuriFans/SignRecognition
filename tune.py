"""tune.py - Fine-tune LeNet5

This script fine-tunes the LeNet model on the project's training data.
Features:
 - adjustable learning rate, dropout rate, weight decay
 - optional warm-start from a checkpoint
 - early stopping based on validation accuracy (patience + min_delta)
 - run all 4 presets automatically with: python tune.py

Usage examples:
  python tune.py                 # Run all 4 presets sequentially
  python tune.py --preset balanced --epochs 20
  python tune.py --ckpt trained_models/best_lenet_basic.pth --epochs 60 --lr 0.0003
  python tune.py --eval-only --ckpt trained_models/best_lenet_none.pth
  python tune.py --preset default_advanced_augmentation
The script writes the best checkpoint to `trained_models/best_lenet_tuned_<params>.pth`.
"""

import argparse
import os
import time
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import create_model
from dataset import GTSRBDataset
from augmentation import get_augmentation_transforms


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		# step scheduler per batch if provided (e.g., OneCycleLR)
		if scheduler is not None:
			try:
				scheduler.step()
			except Exception:
				pass

		running_loss += loss.item() * images.size(0)
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()

	avg_loss = running_loss / total if total > 0 else 0.0
	acc = 100.0 * correct / total if total > 0 else 0.0
	return avg_loss, acc


def validate_epoch(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			running_loss += loss.item() * images.size(0)
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

	avg_loss = running_loss / total if total > 0 else 0.0
	acc = 100.0 * correct / total if total > 0 else 0.0
	return avg_loss, acc


def load_checkpoint_weights(model, ckpt_path, device):
	if not os.path.exists(ckpt_path):
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
	checkpoint = torch.load(ckpt_path, map_location=device)
	# Accept both wrapped checkpoints and raw state_dict
	state = None
	if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
		state = checkpoint['model_state_dict']
	elif isinstance(checkpoint, dict) and any(k.startswith('layer') or k in checkpoint for k in checkpoint):
		# could be other metadata-wrapped dict, try to use as state dict
		state = checkpoint
	else:
		state = checkpoint

	try:
		model.load_state_dict(state)
	except RuntimeError:
		# try non-strict
		model.load_state_dict(state, strict=False)


def tune_single_preset(args, device):
	"""Run fine-tuning for a single preset configuration."""
	# Transforms and dataset
	train_transform, val_transform, _ = get_augmentation_transforms(args.aug, input_size=48)
	dataset_full = GTSRBDataset(root_dir='Final_Training', transform=train_transform, is_train=True)
	# 80/20 split
	train_size = int(0.8 * len(dataset_full))
	val_size = len(dataset_full) - train_size
	train_dataset, val_dataset = torch.utils.data.random_split(dataset_full, [train_size, val_size])
	# ensure validation uses val_transform
	val_dataset.dataset.transform = val_transform

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type=='cuda'))
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=='cuda'))

	# Model
	model = create_model('lenet', num_classes=43, dropout_rate=args.dropout).to(device)

	if args.ckpt:
		print(f"Loading weights from checkpoint: {args.ckpt}")
		try:
			load_checkpoint_weights(model, args.ckpt, device)
			print("Checkpoint loaded (weights only).")
		except Exception as e:
			print(f"Warning: could not load checkpoint weights: {e}")

	# Evaluate-only mode
	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	if args.eval_only:
		val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
		print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
		return

	# Optimizer selection
	if args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif args.optimizer == 'adamw':
		optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif args.optimizer == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	# Scheduler (e.g., OneCycleLR) - stepped per batch inside train_epoch
	scheduler = None
	if args.scheduler == 'onecycle':
		try:
			# prefer user-provided max_lr, otherwise use args.lr
			max_lr = args.max_lr if args.max_lr is not None else args.lr
			from torch.optim.lr_scheduler import OneCycleLR
			# steps_per_epoch must be > 0
			steps_per_epoch = max(1, len(train_loader))
			scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)
		except Exception as e:
			print(f"Warning: could not create OneCycleLR scheduler: {e}")
			scheduler = None

	best_val = -1.0
	epochs_no_improve = 0
	best_checkpoint = None

	for epoch in range(1, args.epochs + 1):
		print(f"\nEpoch [{epoch}/{args.epochs}]")
		train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler=scheduler)
		val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

		print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
		print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

		# Early stopping check (on validation accuracy)
		improved = (val_acc - best_val) > args.min_delta
		if improved:
			best_val = val_acc
			epochs_no_improve = 0
			
			# Build descriptive filename
			lr_s = f"{args.lr:g}"
			bs_s = str(args.batch_size)
			drop_s = f"{args.dropout:g}"
			wd_s = f"{args.weight_decay:g}"
			preset_s = args.preset
			
			param_parts = [f"lr{lr_s}", f"bs{bs_s}", f"drop{drop_s}", f"wd{wd_s}", f"preset{preset_s}"]
			param_summary = '_'.join(param_parts)
			
			# Save to tuned_models (for tracking)
			os.makedirs('tuned_models', exist_ok=True)
			tuned_path = os.path.join('tuned_models', f'tuned_lenet_{param_summary}.pth')
			
			checkpoint_data = {
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'accuracy': best_val,
				'augmentation': args.aug,
				'model': 'lenet',
				'input_size': 48,
				'lr': args.lr,
				'dropout': args.dropout,
				'weight_decay': args.weight_decay,
				'batch_size': args.batch_size,
				'preset': args.preset,
			}
			
			torch.save(checkpoint_data, tuned_path)
			print(f"Checkpoint saved to: {tuned_path}")
			
			# IMMEDIATELY copy to trained_models with descriptive name
			os.makedirs('trained_models', exist_ok=True)
			trained_path = os.path.join('trained_models', f'best_lenet_tuned_{param_summary}.pth')
			
			try:
				torch.save(checkpoint_data, trained_path)
				best_checkpoint = trained_path
				print(f"✓ Best model copied to: {trained_path} (Val Acc: {best_val:.2f}%)")
			except Exception as e:
				print(f"✗ Failed to copy to trained_models: {e}")
				best_checkpoint = tuned_path
		else:
			epochs_no_improve += 1
			print(f"No improvement for {epochs_no_improve} epoch(s)")

			if epochs_no_improve >= args.patience:
				print(f"Early stopping triggered (patience={args.patience})")
				break

	print("\nFine-tuning completed.")
	if best_checkpoint:
		print(f"Best checkpoint (this run): {best_checkpoint} (Val Acc: {best_val:.2f}%)")
	else:
		print("No improvement observed during this run; check tuned_models for previous runs.")

	# After finishing (or even if no improvement this run), select the best tuned model among tuned_models
	tuned_dir = 'tuned_models'
	trained_dir = 'trained_models'
	os.makedirs(trained_dir, exist_ok=True)
	os.makedirs('records', exist_ok=True)

	best_overall = None
	best_overall_acc = -1.0
	tuned_files = []
	if os.path.isdir(tuned_dir):
		# Walk subdirectories so that runs saved in per-parameter folders are discovered
		for root, _, files in os.walk(tuned_dir):
			for fn in files:
				if fn.endswith('.pth') and 'tuned_lenet' in fn:
					full = os.path.join(root, fn)
					try:
						ck = torch.load(full, map_location='cpu')
						acc = None
						if isinstance(ck, dict):
							acc = ck.get('accuracy')
						tuned_files.append({'path': full, 'accuracy': acc, 'meta': ck, 'run_dir': os.path.basename(root)})
						if isinstance(acc, (int, float)) and acc > best_overall_acc:
							best_overall_acc = acc
							best_overall = full
					except Exception:
						tuned_files.append({'path': full, 'accuracy': None, 'meta': None, 'run_dir': os.path.basename(root)})

	# If we found a best overall, copy/rename into trained_models as the single best
	copied_path = None
	if best_overall:
		# Try to create a descriptive filename based on saved metadata (lr, batch_size, dropout, weight_decay)
		best_meta = None
		for entry in tuned_files:
			if entry.get('path') == best_overall:
				best_meta = entry.get('meta')
				break

		# Helper to render model name nicely
		def _pretty_model_name(raw_name: str):
			if not raw_name:
				return 'Model'
			m = raw_name.lower()
			mapping = {
				'lenet': 'LeNet',
				'minivgg': 'MiniVGG',
				'mobilenetv2_025': 'MobileNetV2_025',
				'mobilenetv4_small': 'MobileNetV4_small'
			}
			return mapping.get(m, raw_name)

		def _fmt_val(v):
			if v is None:
				return None
			if isinstance(v, float):
				# use general format (compact) for floats
				return f"{v:g}"
			return str(v)

		param_parts = []
		# Default to lowercase model key for the filename (e.g., 'lenet')
		model_key = 'lenet'
		if isinstance(best_meta, dict):
			model_key = str(best_meta.get('model', 'lenet')).lower()
			lr_s = _fmt_val(best_meta.get('lr'))
			bs_s = _fmt_val(best_meta.get('batch_size'))
			drop_s = _fmt_val(best_meta.get('dropout'))
			wd_s = _fmt_val(best_meta.get('weight_decay'))
			preset_s = _fmt_val(best_meta.get('preset'))
			if lr_s:
				param_parts.append(f"lr{lr_s}")
			if bs_s:
				param_parts.append(f"bs{bs_s}")
			if drop_s:
				param_parts.append(f"drop{drop_s}")
			if wd_s:
				param_parts.append(f"wd{wd_s}")
			if preset_s:
				# include preset last if present
				param_parts.append(f"preset{preset_s}")

		param_summary = '_'.join(param_parts)
		if param_summary:
			best_name = f"best_{model_key}_tuned_{param_summary}.pth"
		else:
			# fallback to timestamped name if no params available
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			best_name = f"best_lenet_tuned_{timestamp}.pth"

		dest = os.path.join(trained_dir, best_name)
		try:
			shutil.copy2(best_overall, dest)
			copied_path = dest
			print(f"Selected best tuned model: {best_overall} -> {dest} (Acc: {best_overall_acc:.2f}%)")
		except Exception as e:
			print(f"Failed to copy best tuned model: {e}")
	else:
		print("No tuned models found in tuned_models/ to select as best.")

	# Write a summary table to records/tune_results.txt
	summary_path = os.path.join('records', 'tune_results.txt')
	lines = []
	header = f"Tune Results Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
	lines.append(header)
	lines.append(f"{'File':<60} {'Accuracy':>10} {'Selected':>10}")
	lines.append('-' * 90)
	for entry in tuned_files:
		path = entry.get('path')
		acc = entry.get('accuracy')
		acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else 'N/A'
		sel = 'YES' if path == best_overall else ''
		lines.append(f"{os.path.basename(path):<60} {acc_str:>10} {sel:>10}")

	if copied_path:
		lines.append('\n')
		lines.append(f"Copied best tuned model to: {copied_path}")

	with open(summary_path, 'w') as f:
		f.write('\n'.join(lines) + '\n')
	print(f"Tune summary written to: {summary_path}")


def main():
	parser = argparse.ArgumentParser(description='Fine-tune LeNet')
	parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint to warm-start from')
	parser.add_argument('--preset', type=str, default=None, choices=['fast', 'balanced', 'conservative', 'default'],
						help='Pre-defined tuning preset (if None, all presets will be run)')
	# If user does not provide explicit numeric values we detect None and apply the preset values below.
	parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overridden by preset if not provided)')
	parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overridden by preset if not provided)')
	parser.add_argument('--lr', type=float, default=None, help='Learning rate (overridden by preset if not provided)')
	parser.add_argument('--dropout', type=float, default=None, help='Dropout rate (overridden by preset if not provided)')
	parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (overridden by preset if not provided)')
	parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (epochs)')
	parser.add_argument('--min_delta', type=float, default=None, help='Minimum change to qualify as improvement')
	parser.add_argument('--eval-only', action='store_true', help='Only evaluate the provided checkpoint')
	parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','adamw','sgd'], help='Optimizer to use for fine-tuning')
	parser.add_argument('--scheduler', type=str, default='none', choices=['none','onecycle'], help='LR scheduler to use (onecycle recommended for sgd)')
	parser.add_argument('--max_lr', type=float, default=None, help='Max LR for OneCycleLR (optional)')
	parser.add_argument('--aug', type=str, default='basic', choices=['none', 'basic', 'advanced'], help='Augmentation strategy for fine-tuning')
	args = parser.parse_args()

	# Pre-defined presets (defaults for runs without explicit numeric params)
# Pre-defined presets (defaults for runs without explicit numeric params)
	presets = {
		'fast': {
			'epochs': 15,
			'batch_size': 128,
			'lr': 3e-4,
			'dropout': 0.5,
			'weight_decay': 1e-4,
			'patience': 5,
			'min_delta': 0.0,
		},
		'balanced': {
			'epochs': 40,
			'batch_size': 128,
			'lr': 2e-4,
			'dropout': 0.55,
			'weight_decay': 2e-4,
			'patience': 10,
			'min_delta': 0.0,
		},
		'conservative': {
			'epochs': 60,
			'batch_size': 64,
			'lr': 1e-4,
			'dropout': 0.6,
			'weight_decay': 3e-4,
			'patience': 15,
			'min_delta': 0.0,
		},
		'default': {
			'epochs': 50,
			'batch_size': 128,
			'lr': 2.5e-4,
			'dropout': 0.55,
			'weight_decay': 2.5e-4,
			'patience': 12,
			'min_delta': 0.0,
		},
	}
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# If preset is None and no explicit parameters provided, run all presets
	if args.preset is None and args.epochs is None and args.batch_size is None and args.lr is None:
		print("\n" + "="*60)
		print("Running all 5 presets sequentially...")
		print("="*60 + "\n")
		for preset_name in ['fast', 'balanced', 'conservative', 'default']:
			print(f"\n{'#'*60}")
			print(f"Starting preset: {preset_name.upper()}")
			print(f"{'#'*60}\n")
			# Create a copy of args and set the preset
			args_copy = argparse.Namespace(**vars(args))
			args_copy.preset = preset_name
			# Apply preset values
			preset_values = presets[preset_name]
			for key, val in preset_values.items():
				setattr(args_copy, key, val)
			# Run tuning for this preset
			tune_single_preset(args_copy, device)
		return
	
	# If preset is None but some parameters are provided, use balanced as default
	if args.preset is None:
		args.preset = 'balanced'
	
	# Apply preset values only for the parameters not explicitly provided (i.e., still None)
	preset_values = presets.get(args.preset, presets['balanced'])
	for key, val in preset_values.items():
		if getattr(args, key) is None:
			setattr(args, key, val)
	
	# Run single preset
	tune_single_preset(args, device)


if __name__ == '__main__':
	main()
