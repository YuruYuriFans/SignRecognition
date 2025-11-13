"""
ablation.py - Simple ablation sweep runner

This script uses the `build_lenet_variant` helper in `model.py` to create
LeNet variants and run short (1-2 epoch) training runs to compare them.

Outputs:
 - ablated_models/<variant_run>/ablation_ck_epochN.pth
 - optionally copies top-N best checkpoints into trained_models/
 - CSV summary: records/ablation_results.csv

Usage (quick):
	python3 ablation.py --epochs 1

"""

import os
import csv
import time
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Try optional profiling libs
try:
	from thop import profile as thop_profile
	_THOP = True
except Exception:
	thop_profile = None
	_THOP = False

from model import build_lenet_variant
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


def _fmt_val(v):
	if v is None:
		return None
	if isinstance(v, float):
		return f"{v:g}"
	return str(v)


def safe_name(s: str) -> str:
	return str(s).replace(' ', '_').replace('.', 'p')


def measure_model_costs(model, device):
	info = {}
	# params
	try:
		num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		info['num_parameters'] = int(num_params)
		total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
		info['model_size_bytes'] = int(total_bytes)
	except Exception:
		info['num_parameters'] = None
		info['model_size_bytes'] = None

	# FLOPs (MACs) via thop if available
	if _THOP:
		try:
			dummy = torch.randn(1, 3, 48, 48).to(device)
			macs, params = thop_profile(model, inputs=(dummy,), verbose=False)
			info['flops_macs'] = float(macs)
		except Exception:
			info['flops_macs'] = None
	else:
		info['flops_macs'] = None

	return info


def run_ablation(configs, epochs=1, batch_size=64, lr=1e-4, export_top=1, aug='basic'):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	# Data
	train_transform, val_transform, _ = get_augmentation_transforms(aug, input_size=48)
	dataset_full = GTSRBDataset(root_dir='Final_Training', transform=train_transform, is_train=True)
	if len(dataset_full) == 0:
		raise RuntimeError('Training dataset is empty or not found in Final_Training/')

	# 90/10 split
	train_size = int(0.9 * len(dataset_full))
	val_size = len(dataset_full) - train_size
	train_dataset, val_dataset = torch.utils.data.random_split(dataset_full, [train_size, val_size])
	val_dataset.dataset.transform = val_transform

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type=='cuda'))
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type=='cuda'))

	os.makedirs('ablated_models', exist_ok=True)
	os.makedirs('trained_models', exist_ok=True)
	os.makedirs('records', exist_ok=True)

	results = []

	for cfg in configs:
		name = cfg.get('name') or cfg.get('id') or f"variant_{int(time.time())}"
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

		# Build a descriptive run folder: model + preset/manual + params
		lr_s = _fmt_val(cfg.get('lr', lr))
		bs_s = _fmt_val(cfg.get('batch_size', batch_size))
		drop_s = _fmt_val(cfg.get('dropout', cfg.get('dropout', 0.5)))
		preset_s = cfg.get('preset', 'manual')
		model_label = cfg.get('model_label', 'lenet')

		param_parts = [f"{safe_name(model_label)}", f"preset_{safe_name(preset_s)}"]
		if lr_s:
			param_parts.append(f"lr{lr_s}")
		if bs_s:
			param_parts.append(f"bs{bs_s}")
		if drop_s:
			param_parts.append(f"drop{drop_s}")
		run_folder = '_'.join(param_parts)
		run_dir = os.path.join('ablated_models', run_folder + '_' + name + '_' + timestamp)
		os.makedirs(run_dir, exist_ok=True)

		print(f"\n=== Running ablation: {name} -> {run_dir} ===")

		# Instantiate model
		model = build_lenet_variant(
			num_conv_layers=cfg.get('num_conv_layers', 3),
			conv_channels=cfg.get('conv_channels', None),
			kernel_sizes=cfg.get('kernel_sizes', None),
			fc_sizes=cfg.get('fc_sizes', None),
			activation=cfg.get('activation', 'relu'),
			dropout=cfg.get('dropout', 0.5),
			num_classes=cfg.get('num_classes', 43)
		).to(device)

		optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr', lr), weight_decay=cfg.get('weight_decay', 0.0))
		criterion = nn.CrossEntropyLoss()

		best_val = -1.0
		best_ck = None
		last_val = None

		model_costs = measure_model_costs(model, device)

		for epoch in range(1, epochs + 1):
			print(f"Epoch [{epoch}/{epochs}]")
			t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
			v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
			print(f" Train Acc: {t_acc:.2f}%, Val Acc: {v_acc:.2f}%")

			# Save checkpoint in run dir
			ck_path = os.path.join(run_dir, f'ablation_ck_epoch{epoch}.pth')
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'accuracy': v_acc,
				'config': cfg,
			}, ck_path)

			if v_acc > best_val:
				best_val = v_acc
				best_ck = ck_path

			last_val = v_acc

		# Copy best into trained_models if requested
		copied = None
		if best_ck and export_top > 0:
			try:
				best_name = f"best_lenet_ablation_{run_folder}_{timestamp}.pth"
				dest = os.path.join('trained_models', best_name)
				torch.save(torch.load(best_ck, map_location='cpu'), dest)
				copied = dest
				print(f"Copied best to trained_models: {dest}")
			except Exception as e:
				print(f"Failed to copy to trained_models: {e}")

		# Record result row
		results.append({
			'variant': name,
			'run_dir': run_dir,
			'best_val_acc': best_val if best_val is not None else 'N/A',
			'last_val_acc': last_val if last_val is not None else 'N/A',
			'num_parameters': model_costs.get('num_parameters'),
			'model_size_MB': round(model_costs.get('model_size_bytes', 0) / (1024**2), 3) if model_costs.get('model_size_bytes') else 'N/A',
			'flops_macs': model_costs.get('flops_macs', 'N/A'),
			'copied_to_trained_models': copied or '',
			'timestamp': timestamp
		})

	# Write CSV summary
	csv_path = os.path.join('records', 'ablation_results.csv')
	fieldnames = ['variant', 'run_dir', 'best_val_acc', 'last_val_acc', 'num_parameters', 'model_size_MB', 'flops_macs', 'copied_to_trained_models', 'timestamp']
	write_header = not os.path.exists(csv_path)
	with open(csv_path, 'a', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		if write_header:
			writer.writeheader()
		for row in results:
			writer.writerow(row)

	print(f"\nAblation sweep complete. Summary written to: {csv_path}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run quick ablation experiments')
	parser.add_argument('--epochs', type=int, default=1, help='Epochs per run (quick test default:1)')
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--export-top', type=int, default=1, help='How many best checkpoints to copy into trained_models per variant (0 to disable)')
	parser.add_argument('--aug', type=str, default='basic', choices=['none','basic','advanced'])
	args = parser.parse_args()

	# Minimal example configurations: remove second conv, change kernels to 3x3, use leakyrelu
	configs = [
		{
			'name': 'remove_second_conv',
			'num_conv_layers': 2,
			'conv_channels': [16, 32],
			'kernel_sizes': [3, 3],
			'activation': 'relu',
			'dropout': 0.5,
		},
		{
			'name': 'kernels_3x3',
			'num_conv_layers': 3,
			'conv_channels': [16, 32, 64],
			'kernel_sizes': [3, 3, 3],
			'activation': 'relu',
			'dropout': 0.5,
		},
		{
			'name': 'leakyrelu',
			'num_conv_layers': 3,
			'conv_channels': [16, 32, 64],
			'kernel_sizes': [3, 3, 3],
			'activation': 'leakyrelu',
			'dropout': 0.5,
		}
		,
		{
			'name': 'elu',
			'num_conv_layers': 3,
			'conv_channels': [16, 32, 64],
			'kernel_sizes': [3, 3, 3],
			'activation': 'elu',
			'dropout': 0.5,
		},
		{
			'name': 'tanh',
			'num_conv_layers': 3,
			'conv_channels': [16, 32, 64],
			'kernel_sizes': [3, 3, 3],
			'activation': 'tanh',
			'dropout': 0.5,
		},		{
			'name': 'kernel_5x5_all',
			'num_conv_layers': 3,
			'conv_channels': [16, 32, 64],
			'kernel_sizes': [5, 5, 5],
			'activation': 'relu',
			'dropout': 0.5,
		},
		{
			'name': 'kernel_7x7_first',
			'num_conv_layers': 3,
			'conv_channels': [16, 32, 64],
			'kernel_sizes': [7, 3, 3],
			'activation': 'relu',
			'dropout': 0.5,
		},		
		{
			'name': 'act_selu',
			'num_conv_layers': 3,
			'conv_channels': [16, 32, 64],
			'kernel_sizes': [3, 3, 3],
			'activation': 'selu',
			'dropout': 0.5,
		},
		### Kernel sizes 
		{
			'name': 'kernel_mixed_531',
			'num_conv_layers': 3,
			'conv_channels': [16, 32, 64],
			'kernel_sizes': [5, 3, 1],
			'activation': 'relu',
			'dropout': 0.5,
		},


	]

	run_ablation(configs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, export_top=args.export_top, aug=args.aug)