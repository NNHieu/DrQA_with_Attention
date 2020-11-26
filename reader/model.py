import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy

# from .config import override_model_args
# from .rnn_reader import RnnDocReader

class Reader(object):

    def __init__(self, wordemb_model):
        self.wordemb_model = wordemb_model

    def expand_dictionary(self, corpus):
        self.wordemb_model.build_vocab(corpus, update=True)

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    # --------------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------------
    def 
