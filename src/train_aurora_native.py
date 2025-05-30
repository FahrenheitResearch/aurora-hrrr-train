#!/usr/bin/env python3
"""
Train native resolution Aurora model on H100
Full 1.3B parameter model with complete HRRR convective variables
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import get_config, update_config_from_args, print_config
from utils import setup_logging, Timer, GPUMonitor, SystemMonitor, save_checkpoint, ensure_directories
from hrrr_dataset import HRRRNativeDataset, create_dataloader
from aurora_native import create_aurora_native

class AuroraNativeTrainer:
    """Trainer for native resolution Aurora on H100"""
    
    def __init__(self, args):
        self.args = args
        self.config = update_config_from_args(args)
        self.logger = logging.getLogger(__name__)
        
        # Setup monitoring
        self.gpu_monitor = GPUMonitor()
        self.system_monitor = SystemMonitor()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
        # Setup directories
        ensure_directories(
            self.config["system"].model_dir,
            self.config["system"].log_dir,
            self.config["system"].results_dir
        )
    
    def setup_model_and_optimizer(self):
        """Setup model, optimizer, and scheduler"""
        
        # Create model
        self.logger.info("🌌 Creating Aurora native model...")
        self.model = create_aurora_native(
            pretrained=not self.args.no_pretrained,
            device=self.config["system"].device
        )
        
        # Get optimizer configuration
        training_config = self.model.configure_for_training()
        
        # Create optimizer
        optimizer_params = training_config["optimizer_params"]
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.config["training"].learning_rate,
            weight_decay=self.config["training"].weight_decay
        )
        
        # Create scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config["training"].warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["training"].max_epochs * len(self.train_loader) - self.config["training"].warmup_steps
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config["training"].warmup_steps]
        )
        
        # Mixed precision scaler
        if self.config["model"].use_mixed_precision:
            self.scaler = GradScaler()
            self.logger.info("✅ Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Log model info
        memory_info = self.model.get_memory_usage()
        self.logger.info("💾 Model memory estimates:")
        for key, value in memory_info.items():
            self.logger.info(f"  {key}: {value:.1f}GB")
    
    def setup_data(self):
        """Setup data loaders"""
        
        self.logger.info("📊 Setting up HRRR dataset...")
        
        # Create dataset
        self.dataset = HRRRNativeDataset(
            data_dir=self.config["system"].data_dir,
            normalize=True,
            cache_static=True
        )
        
        if len(self.dataset) == 0:
            raise ValueError(f"No data found in {self.config['system'].data_dir}")
        
        # Split dataset (simple train/val split)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config["training"].batch_size,
            shuffle=True,
            num_workers=self.config["system"].num_workers if not self.args.debug else 0,
            pin_memory=self.config["training"].pin_memory
        )
        
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=1,  # Always 1 for validation
            shuffle=False,
            num_workers=self.config["system"].num_workers if not self.args.debug else 0,
            pin_memory=self.config["training"].pin_memory
        )
        
        self.logger.info(f"📈 Data setup complete:")
        self.logger.info(f"  Total samples: {len(self.dataset)}")
        self.logger.info(f"  Training: {len(train_dataset)}")
        self.logger.info(f"  Validation: {len(val_dataset)}")
        self.logger.info(f"  Batch size: {self.config['training'].batch_size}")
        self.logger.info(f"  Accumulation steps: {self.config['training'].accumulation_steps}")
        self.logger.info(f"  Effective batch size: {self.config['training'].batch_size * self.config['training'].accumulation_steps}")
    
    def calculate_loss(self, predictions, targets, var_weights=None):
        """Calculate weighted loss across all variables"""
        
        total_loss = 0.0
        var_losses = {}
        
        # Default weights (emphasize convective variables)
        if var_weights is None:
            var_weights = {
                "2t": 1.0, "10u": 1.0, "10v": 1.0, "msl": 1.0,
                "cape": 2.0, "cin": 2.0, "refc": 3.0, "hlcy": 2.0, "mxuphl": 3.0
            }
        
        # Surface variable losses
        for var_name in self.config["model"].surf_vars:
            if var_name in predictions.surf_vars and var_name in targets.surf_vars:
                pred = predictions.surf_vars[var_name]
                target = targets.surf_vars[var_name]
                
                # L2 loss
                var_loss = nn.functional.mse_loss(pred, target)
                weight = var_weights.get(var_name, 1.0)
                
                var_losses[var_name] = var_loss.item()
                total_loss += weight * var_loss
        
        return total_loss, var_losses
    
    def train_step(self, batch):
        """Single training step"""
        
        # Move batch to device
        self.move_batch_to_device(batch)
        
        # Forward pass
        if self.scaler is not None:
            with autocast():
                predictions = self.model(batch)
                loss, var_losses = self.calculate_loss(predictions, batch)
                loss = loss / self.config["training"].accumulation_steps
        else:
            predictions = self.model(batch)
            loss, var_losses = self.calculate_loss(predictions, batch)
            loss = loss / self.config["training"].accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config["training"].accumulation_steps, var_losses
    
    def validation_step(self):
        """Validation loop"""
        
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                self.move_batch_to_device(batch)
                
                if self.scaler is not None:
                    with autocast():
                        predictions = self.model(batch)
                        loss, _ = self.calculate_loss(predictions, batch)
                else:
                    predictions = self.model(batch)
                    loss, _ = self.calculate_loss(predictions, batch)
                
                total_loss += loss.item()
                num_samples += 1
                
                if num_samples >= 10:  # Limit validation samples for speed
                    break
        
        self.model.train()
        return total_loss / num_samples if num_samples > 0 else float('inf')
    
    def move_batch_to_device(self, batch):
        """Move batch to GPU"""
        device = self.config["system"].device
        
        # Move surface variables
        for key in batch.surf_vars:
            if not batch.surf_vars[key].is_cuda:
                batch.surf_vars[key] = batch.surf_vars[key].to(device, non_blocking=True)
        
        # Move atmospheric variables
        for key in batch.atmos_vars:
            if not batch.atmos_vars[key].is_cuda:
                batch.atmos_vars[key] = batch.atmos_vars[key].to(device, non_blocking=True)
        
        # Move static variables
        for key in batch.static_vars:
            if not batch.static_vars[key].is_cuda:
                batch.static_vars[key] = batch.static_vars[key].to(device, non_blocking=True)
        
        # Move metadata
        for attr in ['lat', 'lon', 'time', 'atmos_levels']:
            tensor = getattr(batch.metadata, attr)
            if not tensor.is_cuda:
                setattr(batch.metadata, attr, tensor.to(device, non_blocking=True))
    
    def train_epoch(self):
        """Train one epoch"""
        
        self.model.train()
        epoch_loss = 0.0
        num_steps = 0
        
        for step, batch in enumerate(self.train_loader):
            
            # Training step
            step_loss, var_losses = self.train_step(batch)
            epoch_loss += step_loss
            num_steps += 1
            
            # Gradient accumulation
            if (step + 1) % self.config["training"].accumulation_steps == 0:
                
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["training"].gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"].gradient_clip_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.step += 1
                
                # Logging
                if self.step % self.config["training"].log_every_n_steps == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f"Epoch {self.epoch}, Step {self.step}: "
                        f"Loss={step_loss:.6f}, LR={lr:.2e}"
                    )
                    
                    # Log variable losses
                    var_loss_str = ", ".join([f"{k}={v:.4f}" for k, v in var_losses.items()])
                    self.logger.debug(f"Variable losses: {var_loss_str}")
                    
                    # Monitor memory
                    self.gpu_monitor.log_memory_usage("Training step")
            
            # Early stopping for testing
            if self.args.debug and step >= 5:
                break
        
        return epoch_loss / num_steps if num_steps > 0 else float('inf')
    
    def train(self):
        """Main training loop"""
        
        self.logger.info(f"🚀 Starting Aurora native training...")
        print_config()
        
        # Setup
        self.setup_data()
        self.setup_model_and_optimizer()
        
        # Initial memory check
        self.gpu_monitor.log_memory_usage("Before training")
        self.system_monitor.log_system_usage()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config["training"].max_epochs):
            self.epoch = epoch
            
            self.logger.info(f"\n=== EPOCH {epoch + 1}/{self.config['training'].max_epochs} ===")
            
            # Train epoch
            with Timer(f"Epoch {epoch + 1}"):
                train_loss = self.train_epoch()
            
            # Validation
            if (epoch + 1) % self.config["training"].eval_every_n_epochs == 0:
                with Timer("Validation"):
                    val_loss = self.validation_step()
                
                self.logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    best_model_path = Path(self.config["system"].model_dir) / "aurora_hrrr_native_best.pt"
                    self.model.save_checkpoint(
                        str(best_model_path),
                        epoch=epoch,
                        loss=val_loss,
                        optimizer_state=self.optimizer.state_dict()
                    )
                    self.logger.info(f"✅ New best model saved: {best_model_path}")
            else:
                self.logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config["training"].save_every_n_epochs == 0:
                checkpoint_path = Path(self.config["system"].model_dir) / f"aurora_hrrr_native_epoch_{epoch + 1}.pt"
                self.model.save_checkpoint(
                    str(checkpoint_path),
                    epoch=epoch,
                    loss=train_loss,
                    optimizer_state=self.optimizer.state_dict()
                )
            
            # Monitor resources
            self.gpu_monitor.log_memory_usage(f"End of epoch {epoch + 1}")
            self.system_monitor.log_system_usage()
            
            # Check memory limit
            if not self.gpu_monitor.check_memory_limit(self.config["system"].max_memory_gb):
                self.logger.warning("Approaching memory limit!")
        
        # Final model save
        final_model_path = Path(self.config["system"].model_dir) / "aurora_hrrr_native_final.pt"
        self.model.save_checkpoint(
            str(final_model_path),
            epoch=self.epoch,
            loss=train_loss,
            optimizer_state=self.optimizer.state_dict()
        )
        
        # Training summary
        total_time = time.time() - start_time
        self.logger.info(f"\n🎉 Training completed!")
        self.logger.info(f"Total time: {total_time / 3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")
        self.logger.info(f"Final model: {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Aurora native resolution model on H100")
    
    # Data arguments
    parser.add_argument("--data-dir", default="hrrr_data", help="HRRR data directory")
    parser.add_argument("--model-dir", default="models", help="Model output directory")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (recommend 1 for native res)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained Aurora weights")
    
    # System arguments
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Debug and testing
    parser.add_argument("--debug", action="store_true", help="Debug mode (few steps per epoch)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires GPU.")
        sys.exit(1)
    
    # Check for H100
    gpu_name = torch.cuda.get_device_name()
    if "H100" not in gpu_name:
        logger.warning(f"⚠️  Running on {gpu_name}, not H100. Performance may be suboptimal.")
    else:
        logger.info(f"✅ Running on {gpu_name}")
    
    # Create trainer and run
    try:
        trainer = AuroraNativeTrainer(args)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = trainer.model.load_checkpoint(args.resume)
            trainer.epoch = checkpoint.get("epoch", 0)
            trainer.best_loss = checkpoint.get("loss", float('inf'))
        
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info("❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()