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

from readers.DrQA_Attention import data, model as reader_module, layers
from readers.DrQA_Attention import config as cfg

from pytorch_lightning.loggers import TensorBoardLogger
# from torchsummary import summary

class RnnDocReader(nn.Module):
    RNN_UNIT_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Input size to RNN: word emb + question emb + manual features
        context_input_size = args.embedding_dim
        question_input_size = args.embedding_dim

        self.rnn = layers.CustomRNN(
            input_size=context_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_unit_type=self.RNN_UNIT_TYPES[args.rnn_unit_type],
            padding=args.rnn_padding,
            bidirectional=False
        )
        hidden_size = args.hidden_size + args.num_features
        self.encodes = nn.ModuleList()
        for i in range(args.num_encodes):
            self.encodes.append(layers.EncodeModule(hidden_size, hidden_size, args.num_heads))
        
        
        self.pre_context_encodes = layers.EncodeModule(hidden_size, hidden_size, args.num_heads)
        # for i in range(args.pre_context_layers):
        #     self.pre_context_encodes.append(layers.EncodeModule(hidden_size, hidden_size, args.num_heads))
        
        self.context_encodes = layers.EncodeModule(hidden_size, hidden_size, args.num_heads)
        # for i in range(args.context_layers):
        #     self.context_encodes.append(layers.EncodeModule(hidden_size, hidden_size, args.num_heads))
        
        self.qemb_match = layers.EncodeModule(hidden_size, hidden_size, args.num_heads)
        self.ct_head = nn.Linear(hidden_size*2, hidden_size)
        self.cmerge_attn = layers.LinearSeqAttn(hidden_size)
        self.out = nn.Linear(hidden_size, 2)
    


    def forward(self, q_emb, q_mask, t_emb, t_mask, q_f=None, t_f=None):
        """Inputs:
        x1 = context ids             [batch * len_d * embedding_dim]
        x1_f = context features indices  [batch * len_d * nfeat]
        x1_mask = context padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q * embedding_dim]
        x2_mask = question padding mask        [batch * len_q]
        """
        # # Embed both context and question
        # x1_emb = self.embedding(x1)
        # x2_emb = self.embedding(x2)
        question_hiddens = torch.cat((self.rnn(q_emb, q_mask), q_f), dim=-1)
        context_hiddens  = torch.cat((self.rnn(t_emb, t_mask), t_f), dim=-1)
#         context_hiddens = self.rnn(x1_emb.transpose(0, 1))[0].transpose(0, 1)
#         question_hiddens =self.rnn(x2_emb.transpose(0, 1))[0].transpose(0, 1)
        for i in range(self.args.num_encodes):
            question_hiddens = self.encodes[i](question_hiddens, question_hiddens, question_hiddens, q_mask)
            context_hiddens = self.encodes[i](context_hiddens, context_hiddens, context_hiddens, t_mask)
        
        # for i in range(self.args.pre_context_layers):
        context_hiddens = self.pre_context_encodes(context_hiddens, context_hiddens, context_hiddens, t_mask)
            
        context_hiddens_q = self.qemb_match(context_hiddens, question_hiddens, question_hiddens, q_mask)
        context_hiddens_q_cat = torch.cat((context_hiddens, context_hiddens_q), dim=-1)
        context_hiddens = self.ct_head(context_hiddens_q_cat.view(-1, context_hiddens_q_cat.size(-1))).view(context_hiddens_q_cat.size(0), context_hiddens_q_cat.size(1), int(context_hiddens_q_cat.size(2)/2))
        
        # for i in range(self.args.context_layers):
        context_hiddens = self.context_encodes(context_hiddens, context_hiddens, context_hiddens, t_mask)
            
        
        
        context_merge_weights = self.cmerge_attn(context_hiddens, t_mask)
        context_hidden = layers.weighted_avg(context_hiddens, context_merge_weights)
        
#         question_hiddens, context_hiddens = self.att2(question_hiddens, x2_mask, context_hiddens, x1_mask)
#         question_hiddens, context_hiddens = self.att3(question_hiddens, x2_mask, context_hiddens, x1_mask)
        
#         q_merge_weights = self.qmerge_attn(question_hiddens, x2_mask)
#         question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights) # shape(batch, hdim)
#         context_merge_weights = self.context_attn(context_hiddens, question_hidden, x1_mask) # shape(batch, context_len)
#         context_hidden = layers.weighted_avg(context_hiddens, context_merge_weights)
        label_score = self.out(context_hidden)
        return label_score

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
        self.network = RnnDocReader(args, normalize)
        
        self.loss_fct = CrossEntropyLoss(weight=torch.Tensor([0.3, 0.7], device=self.device))


    def forward(self, x1, x1_f, x1_mask, x2, x2_f, x2_mask, *args, **kwargs):
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        xfs = [x1_f, x2_f]
        xfembs = []
        for xf in xfs:
            xfembs.append([])
            for i in range(xf.size(1)):
                xfembs[-1].append(self.embedding_feature[i](xf[:, i, :].squeeze(1)))
            xfembs[-1] = torch.cat((xfembs[-1]), dim=2)
        return self.network(x1_emb, x1_mask, x2_emb, x2_mask, *xfembs)

    def training_step(self, batch, batch_idx):
        logits = self(**batch)
        loss = self.loss_fct(logits.view(-1, self.num_labels), batch['labels'].view(-1))
        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]
        self.logger.experiment.add_scalar("Loss/Train_step", loss.item(), self.global_step)
        return {'loss': loss, "preds": preds, "labels": labels}

    
    def training_epoch_end(self, outputs):
        for name,params in self.named_parameters():
	        self.logger.experiment.add_histogram(name,params,self.current_epoch)
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon, weight_decay=self.hparams.weight_decay)

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