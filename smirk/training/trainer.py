import logging
from tqdm import tqdm
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from smirk.training.losses import calculate_loss

log = logging.getLogger(__name__)


class SurrogateTrainer:
    """
    Trains a surrogate model with long-tail reweighting and expert diversity

    Args:
        model:              The surrogate model to train (multi-expert architecture)
        train_loader:       DataLoader yielding (image, soft_target) batches
        test_loader:        DataLoader for evaluation, or None to skip evaluation
        writer:             TensorBoard SummaryWriter
        topk_weights:       Class weights derived from long-tail label distribution

        epochs:             Number of training epochs
        num_experts         Number of expert classifiers
        lambda_diversity    Diversity loss weighting 
        lambda_ce           Cross entropy loss weighting     
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader],
        writer: SummaryWriter,
        topk_weights: torch.Tensor,
        # Hyperparameters
        epochs: int,
        num_experts: int,
        lambda_diversity: float,
        lambda_ce: float

    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.writer = writer
        self.topk_weights = topk_weights

        self.epochs = epochs
        self.num_experts = num_experts
        self.lambda_diversity = lambda_diversity
        self.lambda_ce = lambda_ce

    def fit(self) -> nn.Module:
        """Run the full training loop"""
        best_acc = 0.0
 
        for epoch in range(self.epochs):
            #log.info(f"Epoch {epoch+1} / {self.epochs}",)
            self.train_epoch(epoch)
 
            if self.test_loader is not None and (epoch % 25 == 0 or epoch == self.epochs - 1):
                acc_top1, _ = self.test_epoch(epoch)
                if acc_top1 > best_acc:
                    best_acc = acc_top1
                    log.info("New best accuracy: %.4f", best_acc)
                    self.on_best(epoch, best_acc)

            self.on_epoch_end(epoch)
 
 
        return self.model
    
    # core training / eval
    def train_epoch(self, epoch: int, log_interval: int = 100):
        self.model.train()
        topk_weights = self.topk_weights.to(self.device)
 
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"Train epoch {epoch}", leave=False)):
            data, target = data.to(self.device), target.to(self.device)

            max_prob_idx = torch.argmax(target, dim=1)
            topk_weight = topk_weights[max_prob_idx]
 
            self.optimizer.zero_grad()
 
            loss, kl_loss, ce_loss, div_loss = calculate_loss(
                model=self.model,
                data=data,
                target=target,
                topk_weight=topk_weight,
                num_experts=self.num_experts,
                lambda_diversity=self.lambda_diversity,
                lambda_ce=self.lambda_ce
            )
 
            loss.backward()
            self.optimizer.step()
 
            if batch_idx % log_interval == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                log.info(
                    "Epoch %d [%d/%d]  KL: %.6f  CE: %.6f  Diversity: %.6f",
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    kl_loss,
                    ce_loss,
                    div_loss,
                )
                self.writer.add_scalar("Loss/train_kl", kl_loss, global_step)
                self.writer.add_scalar("Loss/train_ce", ce_loss, global_step)
                self.writer.add_scalar("Loss/train_diversity", div_loss, global_step)
                self.writer.add_scalar("Loss/train_total", loss.item(), global_step)
 
    def test_epoch(self, epoch: int):
        self.model.eval()
 
        test_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        n = 0
 
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc=f"Test epoch {epoch}", leave=False):
                data, target = data.to(self.device), target.to(self.device)

                if hasattr(self.model, "AuxLogits"):
                    output, *_ = self.model(data, 0)
                else:
                    output, _ = self.model(data)
 
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                n += target.size(0)
 
                _, pred_top1 = output.max(dim=1)
                correct_top1 += pred_top1.eq(target.view_as(pred_top1)).sum().item()
 
                _, pred_top5 = output.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(target.view(-1, 1).expand_as(pred_top5)).sum().item()
 
        test_loss /= n
        acc_top1 = 100.0 * correct_top1 / n
        acc_top5 = 100.0 * correct_top5 / n
 
        log.info(
            "Test — loss: %.4f | Top-1: %d/%d (%.2f%%) | Top-5: %d/%d (%.2f%%)",
            test_loss, correct_top1, n, acc_top1, correct_top5, n, acc_top5,
        )
 
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/top1", acc_top1, epoch)
        self.writer.add_scalar("Accuracy/top5", acc_top5, epoch)
 
        return acc_top1, acc_top5
    
    # checkpointing
    def on_best(self, epoch: int, acc: float):
        pass

    def on_epoch_end(self, epoch: int):
        pass
