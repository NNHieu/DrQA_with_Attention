import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy

# from .config import override_model_args
# from .rnn_reader import RnnDocReader

class Reader(object):

    def __init__(self, args, wordemb_model, normalize=True):
        self.update = 0

        self.wordemb_model = wordemb_model
        self.word_dict = word_dict

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)


    def expand_dictionary(self, corpus):
        self.wordemb_model.build_vocab(corpus, update=True)
        self.wordemb_model.train(corpus)
    
    def load_embeddings(self, wordemb_model):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        self.wordemb_model = wordemb_model
        words = {w for w in words if w in self.wordemb_model.vocab.keys()}
        # logger.info('Loading pre-trained embeddings for %d words from %s' %
        #             (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        for w in self.wordemb_model.vocab:
            if w not in words:
                vec = torch.Tensor(self.wordemb_model[w])
                if w not in vec_counts:
                    vec_counts[w] = 1
                    embedding[self.word_dict[w]].copy_(vec)
                else:
                    logging.warning(
                        'WARN: Duplicate embedding found for %s' % w
                    )
                    vec_counts[w] = vec_counts[w] + 1
                    embedding[self.word_dict[w]].add_(vec)
        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights.
        Inputs:
        ex[0] = context ids                 [batch * len_c]
        ex[1] = context features indices    [batch * len_c * nfeat]
        ex[2] = context padding mask        [batch * len_c]
        ex[3] = question word indices       [batch * len_q]
        ex[4] = question padding mask       [batch * len_q]
        """
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

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
        loss = F.nll_loss(score_correct, target_label)

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

    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
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
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores

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
            score_correct = self.network(*inputs) # shape(batch, context_len)

        # Decode predictions# score_s, score_e = self.network(*inputs)
        # score_s = score_s.data.cpu()
        # score_e = score_e.data.cpu()
        score_correct = score_correct.data.cpu()
        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_correct, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                # return self.decode(*args)
                return score_correct

     @staticmethod
    def decode(score_correct, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.

        Args:
            score_correct: context correct answer score
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score