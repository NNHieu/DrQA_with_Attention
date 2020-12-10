import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
import logging
import copy

# from .config import override_model_args
from .module import RnnDocReader, EmbeddingModule
import logging

logger = logging.getLogger(__name__)

class Reader(object):

    def __init__(self, args, vocab, normalize=True):
        """
        parameters:
        ------------
        args-config:
            model_type

        """
        self.args = args
        self.updates = 0
        self.use_cuda = False
        self.args.vocab_size = len(vocab)
        self.vocab = vocab

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)


    def expand_dictionary(self, words):
        to_add = {self.vocab.normalize(w) for w in words
                  if w not in self.vocab}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.vocab.add(w)
            self.args.vocab_size = len(self.vocab)
            logger.info('New vocab size: %d' % len(self.vocab))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
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
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        for w in wv.vocab:
            if w not in words:
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

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    
    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------
    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    def set_criterion(self, criterion):
        self.criterion = criterion 

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights.
        Inputs:
        ex[0] = context ids                 [batch * len_c]
        ex[1] = context features indices    [batch * len_c * nfeat]
        ex[2] = context padding mask        [batch * len_c]
        ex[3] = question word indices       [batch * len_q]
        ex[4] = question padding mask       [batch * len_q]
        ex[5] = target label                [batch]
        """
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        if not self.criterion:
            raise RuntimeError('No criterior set.')
        # Train mode
        self.network.train()
        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else e.cuda(non_blocking=True)
                      for e in ex[:5]]
            # target_s = ex[5].cuda(non_blocking=True)
            # target_e = ex[6].cuda(non_blocking=True)
            target_label = ex[5].cuda(non_blocking=True)
        else:
            inputs = [e if e is None else e for e in ex[:5]]
            # target_s = ex[5]
            # target_e = ex[6]
            target_label = ex[5]


        # Run forward
        # score_s, score_e = self.network(*inputs)
        score_correct = self.network(*inputs)

        # Compute loss and accuracies
        # loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        loss = self.criterion(score_correct, target_label)
        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                       self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.item(), ex[0].size(0) # loss và batch size

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""
        pass
        # Reset fixed embeddings to original value
        # if self.args.tune_partial > 0:
        #     if self.parallel:
        #         embedding = self.network.module.embedding.weight.data
        #         fixed_embedding = self.network.module.fixed_embedding
        #     else:
        #         embedding = self.network.embedding.weight.data
        #         fixed_embedding = self.network.fixed_embedding

        #     # Embeddings to fix are the last indices
        #     offset = embedding.size(0) - fixed_embedding.size(0)
        #     if offset >= 0:
        #         embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Batch includes:
            ex[0] = context ids                 [batch * len_c]
            ex[1] = context features indices    [batch * len_c * nfeat]
            ex[2] = context padding mask        [batch * len_c]
            ex[3] = question word indices       [batch * len_q]
            ex[4] = question padding mask       [batch * len_q]
        Output:
            pred_label

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else e.cuda(non_blocking=True)
                      for e in ex[:5]]
        else:
            inputs = [e for e in ex[:5]]

        # Run forward
        with torch.no_grad():
            # score_s, score_e = self.network(*inputs)
            label_score = self.network(*inputs) # shape(batch, context_len)

        # Decode predictions# score_s, score_e = self.network(*inputs)
        # score_s = score_s.data.cpu()
        # score_e = score_e.data.cpu()
        label_score = label_score.data.cpu()
        return torch.argmax(label_score, dim=1)

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

class FastTextReader(nn.Module):
    def __init__(self, args, vocab, normalize=True):
        super(FastTextReader, self).__init__()
        self.args = args
        self.args.vocab_size = len(vocab)
        self.vocab = vocab
        
        self.embedding = EmbeddingModule(args, vocab)
        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)
    
    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        return self.network(x1_emb, x1_f, x1_mask, x2_emb, x2_mask)
