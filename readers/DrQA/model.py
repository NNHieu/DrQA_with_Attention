import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.rnn import CustomBRNN
from ..layers.general import LinearSeqAttn, BilinearSeqAttn, uniform_weights, weighted_avg
from ..layers.attn import Multihead, ScaleDotProductAttention, SelfAttnMultihead, EncodeModule
import numpy as np
import logging


class RnnDocReader(nn.Module):
    RNN_UNIT_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args
        # Input size to context RNN: word emb + question emb + manual features
        context_input_size = args.embedding_dim + args.num_features
        # Input size to question RNN: word emb + question emb + manual features
        question_input_size = args.embedding_dim + args.num_features

        # Projection for attention weighted question
        if args.use_qemb:
            if args.num_attn_head == 0:
                self.qemb_match = ScaleDotProductAttention(question_input_size)
            else:
                self.qemb_match = EncodeModule(question_input_size, question_input_size, question_input_size, args.num_attn_head)
            context_input_size += question_input_size

        # RNN context encoder
        self.context_rnn = CustomBRNN(
            input_size=context_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.context_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_unit_type=self.RNN_UNIT_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = CustomBRNN(
            input_size=question_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_unit_type=self.RNN_UNIT_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )
        # Output sizes of rnn encoders
        context_hidden_size = question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            context_hidden_size *= args.context_layers
            question_hidden_size *= args.question_layers
        
        if args.question_merge_self_attn:
            if args.num_attn_head == 0:
                self.question_self_attn = ScaleDotProductAttention(question_hidden_size)
            else:
                self.question_self_attn = EncodeModule(question_hidden_size,question_hidden_size,question_hidden_size,args.num_attn_head)
        # Question merging
        if args.question_merge not in ['avg', 'learn_weight']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        elif args.question_merge == 'learn_weight':
            self.q_merge = LinearSeqAttn(question_hidden_size)


        # Bilinear attention for label
        self.context_attn = BilinearSeqAttn(
            context_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

        self.out = nn.Linear(context_hidden_size, 2)
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
        q = torch.cat((q_emb, q_f), dim=-1)
        
        t = torch.cat((t_emb, t_f), dim=-1)
        if self.args.use_qemb:
            t_q_attn = self.qemb_match(t, q, q, q_mask)
            t = torch.cat((t, t_q_attn), dim=-1)
        
        question_hiddens = self.question_rnn(q, q_mask)
        if self.args.question_merge_self_attn:
            question_hiddens = self.question_self_attn(question_hiddens, question_hiddens, question_hiddens, q_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = uniform_weights(question_hiddens, q_mask)
        elif self.args.question_merge == 'learn_weight':
            q_merge_weights = self.q_merge(question_hiddens, q_mask)
        else:
            raise NotImplementedError('merge_mode = %s' % self.args.question_merge)
        question_hidden = weighted_avg(question_hiddens, q_merge_weights)
        
        context_hiddens = self.context_rnn(t, t_mask)
        t_merge_weights = self.context_attn(context_hiddens, question_hidden, t_mask)
        context_hidden = weighted_avg(context_hiddens, t_merge_weights)
        return self.out(context_hidden)