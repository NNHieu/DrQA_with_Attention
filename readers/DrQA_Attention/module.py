import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers
import numpy as np
import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class EmbeddingModule(nn.Module):
    def __init__(self, args, vocab):
        super(EmbeddingModule, self).__init__()
        self.args = args
        self.vocab = vocab

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=vocab.PAD_index)
        if self.args.layernorm_emb:
            self.layernorm = nn.LayerNorm((args.embedding_dim,), eps=1e-05, elementwise_affine=True)

        if self.args.dropout_emb:
            self.dropout = nn.Dropout(p=self.args.dropout_emb)
        

    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.vocab.normalize(w) for w in words
                  if w not in self.vocab}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.vocab.add(w)
            self.args.vocab_size = len(self.vocab)
            logger.info('New vocab size: %d' % len(self.vocab))

            old_embedding = self.embedding.weight.data
            self.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add


    def load_embeddings(self, words, wv):
        """Load Gensim embedding model
        Args:
            words: iterable of tokens. Only those that are indexed in the
                dictionary are kept.
        """
        words = {w for w in words if w in self.vocab}
        # logger.info('Loading pre-trained embeddings for %d words from %s' %
        #             (len(words), embedding_file))
        embedding = self.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        for w in wv.vocab:
            if w in words:
                vec = torch.Tensor(np.copy(wv[w]))
                if w not in vec_counts:
                    vec_counts[w] = 1
                    embedding[self.vocab[w]].copy_(vec)
                else:
                    logging.warning(
                        'WARN: Duplicate embedding found for %s' % w
                    )
                    vec_counts[w] = vec_counts[w] + 1
                    embedding[self.vocab[w]].add_(vec)
        for w, c in vec_counts.items():
            embedding[self.vocab[w]].div_(c)

        print('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))
    
    def forward(self, ids):
        emb = self.embedding(ids)
        if self.args.layernorm_emb:
            emb = self.layernorm(emb)
        if self.args.dropout_emb:
            emb = self.dropout(emb)
        return emb

class AttentionModule(nn.Module):
    def __init__(self, args, normalize=True):
        super(AttentionModule, self).__init__()
        # Store config
        self.args = args
        hidden_size = args.hidden_size + args.num_features
        # Projection for attention weighted question
        self.self_attn = layers.AttnMatch(hidden_size)
        self.qemb_match = layers.AttnMatch(hidden_size)
        self.ct_head = nn.Conv1d(hidden_size*2, hidden_size, 1)
        

    def forward(self, question_rnn, x2_mask, context_rnn, x1_mask):
        question_hiddens = self.self_attn(question_rnn, question_rnn, x2_mask)
        context_hiddens = self.self_attn(context_rnn, context_rnn, x1_mask)
        context_hiddens_q = self.qemb_match(context_hiddens, question_hiddens, x2_mask)
        context_hiddens = self.ct_head(torch.cat((context_hiddens, context_hiddens_q), dim=-1).transpose(1, 2)).transpose(1, 2)
        return question_hiddens, context_hiddens

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
            num_layers=args.context_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_unit_type=self.RNN_UNIT_TYPES[args.rnn_unit_type],
            padding=args.rnn_padding,
        )
        hidden_size = args.hidden_size + args.num_features
        self.qmerge_attn = layers.LinearSeqAttn(hidden_size)
        
        self.att1 = AttentionModule(args)
        self.att2 = AttentionModule(args)
        self.att3 = AttentionModule(args)

        # Bilinear attention for span start/end
        self.context_attn = layers.BilinearSeqAttn(
            hidden_size,
            hidden_size,
            normalize=normalize,
        )
        self.out = nn.Linear(hidden_size, 2)
    


    def forward(self, x1_emb, x1_mask, x2_emb, x2_mask, x1_f=None, x2_f=None):
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

        context_rnn = torch.cat((self.rnn(x1_emb, x1_mask), x1_f), dim=-1)
        question_rnn = torch.cat((self.rnn(x2_emb, x2_mask), x2_f), dim=-1)
        
        question_hiddens, context_hiddens = self.att1(question_rnn, x2_mask, context_rnn, x1_mask)
        question_hiddens, context_hiddens = self.att2(question_hiddens, x2_mask, context_hiddens, x1_mask)
        question_hiddens, context_hiddens = self.att3(question_hiddens, x2_mask, context_hiddens, x1_mask)
        
        q_merge_weights = self.qmerge_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights) # shape(batch, hdim)
        context_merge_weights = self.context_attn(context_hiddens, question_hidden, x1_mask) # shape(batch, context_len)
        context_hidden = layers.weighted_avg(context_hiddens, context_merge_weights)
        label_score = self.out(context_hidden)
        return label_score