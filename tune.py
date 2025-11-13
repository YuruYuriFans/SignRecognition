"""tune.py - Fine-tune LeNet5

This script fine-tunes the LeNet model on the project's training data.
Features:
 - adjustable learning rate, dropout rate, weight decay
 - optional warm-start from a checkpoint
 - early stopping based on validation accuracy (patience + min_delta)

Usage examples:
  python tune.py --epochs 20 --lr 1e-4 --dropout 0.5 --weight_decay 1e-4 --patience 5 --ckpt trained_models/best_lenet_none.pth
  python tune.py --eval-only --ckpt trained_models/best_lenet_none.pth

The script writes the best checkpoint to `trained_models/tuned_lenet_<timestamp>.pth`.
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


def train_epoch(model, loader, criterion, optimizer, device):
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


def main():
	parser = argparse.ArgumentParser(description='Fine-tune LeNet')
	parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint to warm-start from')
	parser.add_argument('--preset', type=str, default='balanced', choices=['fast', 'balanced', 'conservative'],
						help='Pre-defined tuning preset used when explicit parameters are not provided')
	# If user does not provide explicit numeric values we detect None and apply the preset values below.
	parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overridden by preset if not provided)')
	parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overridden by preset if not provided)')
	parser.add_argument('--lr', type=float, default=None, help='Learning rate (overridden by preset if not provided)')
	parser.add_argument('--dropout', type=float, default=None, help='Dropout rate (overridden by preset if not provided)')
	parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (overridden by preset if not provided)')
	parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (epochs)')
	parser.add_argument('--min_delta', type=float, default=None, help='Minimum change to qualify as improvement')
	parser.add_argument('--eval-only', action='store_true', help='Only evaluate the provided checkpoint')
	parser.add_argument('--aug', type=str, default='basic', choices=['none', 'basic', 'advanced'], help='Augmentation strategy for fine-tuning')
	args = parser.parse_args()

	# Pre-defined presets (defaults for runs without explicit numeric params)
	presets = {
    'fast': {
        'epochs': 5,
        'batch_size': 128,
        'lr': 5e-4,
        'dropout': 0.3,
        'weight_decay': 1e-5,
        'patience': 2,
        'min_delta': 0.0,
    },
    'balanced': {
        'epochs': 20,
        'batch_size': 128,
        'lr': 2e-4,
        'dropout': 0.5,
        'weight_decay': 1e-5,
        'patience': 6,
        'min_delta': 0.001,
    },
    'conservative': {
        'epochs': 50,
        'batch_size': 32,
        'lr': 5e-5,
        'dropout': 0.5,
        'weight_decay': 1e-4,
        'patience': 8,
        'min_delta': 0.01,
    }
}

	# Apply preset values only for the parameters not explicitly provided (i.e., still None)
	preset_values = presets.get(args.preset, presets['balanced'])
	for key, val in preset_values.items():
		if getattr(args, key) is None:
			setattr(args, key, val)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Transforms and dataset
	train_transform, val_transform, _ = get_augmentation_transforms(args.aug, input_size=48)
	dataset_full = GTSRBDataset(root_dir='Final_Training', transform=train_transform, is_train=True)
	# 90/10 split
	train_size = int(0.9 * len(dataset_full))
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
	criterion = nn.CrossEntropyLoss()
	if args.eval_only:
		val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
		print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
		return

	# Optimizer: fine-tune all params by default
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	best_val = -1.0
	epochs_no_improve = 0
	best_checkpoint = None

	for epoch in range(1, args.epochs + 1):
		print(f"\nEpoch [{epoch}/{args.epochs}]")
		train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

		print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
		print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

		# Early stopping check (on validation accuracy)
		improved = (val_acc - best_val) > args.min_delta
		if improved:
			best_val = val_acc
			epochs_no_improve = 0
			# save best
			os.makedirs('tuned_models', exist_ok=True)
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			# Build a run-folder name from hyperparameters so each run is stored in its own subfolder
			def _fmt_val(v):
				if v is None:
					return None
				if isinstance(v, float):
					return f"{v:g}"
				return str(v)

			# Build run folder starting with model name, then preset, then remaining params
			lr_s = _fmt_val(args.lr)
			bs_s = _fmt_val(args.batch_size)
			drop_s = _fmt_val(args.dropout)
			wd_s = _fmt_val(args.weight_decay)
			preset_s = _fmt_val(args.preset)

			# Model label (human-friendly)
			model_label = 'LeNet'

			param_parts = []
			if lr_s:
				param_parts.append(f"lr{lr_s}")
			if bs_s:
				param_parts.append(f"bs{bs_s}")
			if drop_s:
				param_parts.append(f"drop{drop_s}")
			if wd_s:
				param_parts.append(f"wd{wd_s}")

			preset_label = f"preset_{preset_s}" if preset_s else "preset_manual"
			run_folder = f"{model_label}_{preset_label}"
			if param_parts:
				run_folder = run_folder + '_' + '_'.join(param_parts)
			# Create the run directory under tuned_models
			run_dir = os.path.join('tuned_models', run_folder)
			os.makedirs(run_dir, exist_ok=True)

			# Save checkpoint inside the run directory
			save_path = os.path.join(run_dir, f'tuned_lenet_{timestamp}.pth')
			# Persist common hyperparameters in the checkpoint so we can synthesize descriptive filenames later
			torch.save({
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
			}, save_path)
			best_checkpoint = save_path
			print(f"New best model saved to: {save_path}")
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


if __name__ == '__main__':
	main()

