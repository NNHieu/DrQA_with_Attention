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
from transformers import (
    AdamW,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics,
    PhobertTokenizer
)
import datasets
import pandas as pd
from textprocessor.normalization import VncorenlpTokenizer
from gensim.models import KeyedVectors

from readers.DrQA import data, model as reader_module, layers
from readers.DrQA import config as cfg

from pytorch_lightning.loggers import TensorBoardLogger
# from torchsummary import summary

class DrQA(pl.LightningModule):
    def __init__(
        self,
        args,
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
        self.metric = datasets.load_metric(
            'glue',
            'qqp',
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        
        self.args = args
        self.args.vocab_size = len(vocab)
        self.vocab = vocab
        
        self.embedding = reader_module.EmbeddingModule(args, vocab)
        self.embedding_feature = nn.ModuleList()
        for i in range(args.num_features):
            self.embedding_feature.append(nn.Embedding(dim_feature[i], 1, padding_idx=1))
        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        self.network = reader_module.RnnDocReader(args, normalize)
        
        self.loss_fct = CrossEntropyLoss(weight=torch.Tensor([0.3, 0.7], device=self.device))


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
        
        x1_emb = torch.cat((x1_emb, xfembs[0]), dim=2)
        x2_emb = torch.cat((x2_emb, xfembs[1]), dim=2)
        return self.network(x1_emb, x1_mask, x2_emb, x2_mask)

    def training_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        self.logger.experiment.add_scalar("Loss/Train_step", loss.item(), self.global_step)
        return {'loss': loss, "preds": preds, "labels": labels}

    
    def training_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).detach().mean()
        metric = self.metric.compute(predictions=preds, references=labels)
        
        self.logger.experiment.add_scalar("Loss/Train_epoch", loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train", metric['accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("F1/Train", metric['f1'], self.current_epoch)
        

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(**batch)
        val_loss = self.loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).detach().mean()
        metric = self.metric.compute(predictions=preds, references=labels)
#         self.log('val_loss', loss.item(), prog_bar=True)
#         self.log_dict(metric, prog_bar=True)
        
        self.logger.experiment.add_scalar("Loss/Valid", loss, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Valid", metric['accuracy'], self.current_epoch)
        self.logger.experiment.add_scalar("F1/Valid", metric['f1'], self.current_epoch)

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
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

        scheduler = get_linear_schedule_with_warmup(
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
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser