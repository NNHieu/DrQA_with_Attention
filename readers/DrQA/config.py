class Args(object):
    def __init__(self):
        # model architecture
        self.model_type = 'rnn'
        self.embedding_dim = 300
        self.hidden_size = 100
        self.context_layers = 1
        self.question_layers = 1
        self.rnn_unit_type = 'lstm'
        self.concat_rnn_layers = False
        self.question_merge = False
        self.use_qemb = True
        self.rnn_padding = True
        self.question_merge = 'self_attn' # 'avg' hoặc self_attn
        # self.use_in_question 
        # self.use_pos
        # self.use_ner
        # self.use_lemma
        # self.use_tf

        #Optimizer & Training
        self.fix_embeddings = True
        self.optimizer = 'adamax'
        self.learning_rate = 1e-4
        self.momentum = 0
        self.dropout_rnn = 0.4
        self.dropout_rnn_output = False
        # self.dropout_emb = 
        self.grad_clipping = True
        self.max_len = 16
        # self.tune_partial

        self.num_features = 0