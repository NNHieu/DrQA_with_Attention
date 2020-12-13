from argparse import ArgumentParser
from datetime import datetime
from typing import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import pandas as pd
import math
# Config
from .config import add_model_args

#Model
from gensim.models import KeyedVectors
from readers import modules as reader_module
from readers.DrQA.model import RnnDocReader
from readers.DrQA import config as DrQAConfig

#Optim
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# Logger
from pytorch_lightning.loggers import TensorBoardLogger

# Metrics
from pytorch_lightning.metrics.classification import F1
from pytorch_lightning.metrics import Accuracy

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class DrQA(pl.LightningModule):
    def __init__(
        self,
        vocab,
        dim_feature,
        normalize=True,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_labels = 2
        
        self.hparams.vocab_size = len(vocab)
        self.vocab = vocab
        
        # Model
        self.embedding = reader_module.EmbeddingModule(self.hparams, vocab)
        self.embedding_feature = nn.ModuleList()
        for i in range(self.hparams.num_features):
            self.embedding_feature.append(nn.Embedding(dim_feature[i], 1, padding_idx=1))
        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        self.network = RnnDocReader(self.hparams, normalize)
        
        self.loss_fct = CrossEntropyLoss(weight=torch.Tensor([0.3, 0.7], device=self.device))

        # Metrics
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy(compute_on_step=False)
        self.train_f1 = F1(num_classes=2)
        self.valid_f1 = F1(num_classes=2, compute_on_step=False)

    def forward(self, x1, x1_f, x1_mask, x2, x2_f, x2_mask, *args, **kwargs):
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        xfs = [x1_f, x2_f]
        xfembs = []
        for xf in xfs:
            xfembs.append([])
            for i in range(xf.size(1)):
                xfembs[-1].append(self.embedding_feature[i](xf[:, i, :]))
            xfembs[-1] = torch.cat((xfembs[-1]), dim=2)
        return self.network(x1_emb, x1_mask, x2_emb, x2_mask, xfembs[0], xfembs[1])

    def training_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        labels = batch["labels"]
        self.train_acc(logits, labels)
        self.train_f1(logits, labels)
        self.logger.experiment.add_scalar("Loss/Train_step", loss.item(), self.global_step)
        return {'loss': loss}

    
    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).detach().mean()
        self.logger.experiment.add_scalar("Loss/Train_epoch", loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", self.train_acc.compute(), self.current_epoch)
        self.logger.experiment.add_scalar("F1/Train", self.train_f1.compute(), self.current_epoch)
        

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(**batch)
        val_loss = self.loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        labels = batch["labels"]
        self.valid_acc(logits, labels)
        self.valid_f1(logits, labels)
        return {'loss': val_loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).detach().mean()
        self.log('val_loss', loss)
        self.logger.experiment.add_scalar("Loss/Valid", loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Valid", self.valid_acc.compute(), self.current_epoch)
        self.logger.experiment.add_scalar("F1/Valid", self.valid_f1.compute(), self.current_epoch)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(**batch)
        val_loss = self.loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        labels = batch["labels"]
        self.valid_acc(logits, labels)
        self.valid_f1(logits, labels)
        return {'loss': val_loss}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).detach().mean()
        self.logger.experiment.add_scalar("Loss/Test", loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test", self.valid_acc.compute(), self.current_epoch)
        self.logger.experiment.add_scalar("F1/Test", self.valid_f1.compute(), self.current_epoch)

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) 
                // (self.hparams.train_batch_size 
                * max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,  weight_decay=self.hparams.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        return add_model_args(parent_parser)