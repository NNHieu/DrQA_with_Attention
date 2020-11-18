#%%
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from vncorenlp import VnCoreNLP
from src.preprocess.normalization import vcn_word_segment
from src.preprocess.zalo_processor import BertInputFeatures

class RobertaInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 attention_mask,
                 label_id,
                 is_real_example=True):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id
        self.is_real_example = is_real_example
#%%
def convert_feature_to_dataloader(features, batch_size=16, dev_size = 0.2):
    inputs = []
    labels = []
    masks  = []
    pos_neg_samples = [[], []]
    # val_type_ids = []
    for i, feature in tqdm(enumerate(features)):
        pos_neg_samples[feature.label_id].append(i)
        inputs.append(feature.input_ids)
        labels.append(feature.label_id)
        masks.append(feature.attention_mask)
    print()

    num_neg_sample = len(pos_neg_samples[0])
    num_pos_sample = len(pos_neg_samples[1])
    pos_neg_samples = [np.array(index_list) for index_list in pos_neg_samples]
    np.random.shuffle(pos_neg_samples[0])
    np.random.shuffle(pos_neg_samples[1])
    
    train_size = [np.floor(num_neg_sample*(1 - dev_size)),
                    np.floor(num_pos_sample*(1 - dev_size))]
    train_size = [int(i) for i in train_size]
    print("Train size:", train_size)
    print("Val size:", [num_neg_sample - train_size[0], num_pos_sample - train_size[1]])
    train_indexs = list(pos_neg_samples[0][:train_size[0]]) \
                    + list(pos_neg_samples[1][:train_size[1]])
    val_indexs = list(pos_neg_samples[0][train_size[0]:]) \
                    + list(pos_neg_samples[1][train_size[1]:])
    train_inputs = torch.tensor([inputs[i] for i in train_indexs])
    val_inputs = torch.tensor  ([inputs[i] for i in val_indexs]  )
    train_labels = torch.tensor([labels[i] for i in train_indexs])
    val_labels = torch.tensor  ([labels[i] for i in val_indexs]  )
    train_masks = torch.tensor([masks[i] for i in train_indexs])
    val_masks = torch.tensor([masks[i] for i in val_indexs])
    # train_type_ids = torch.tensor(train_type_ids)
    # val_type_ids = torch.tensor(val_type_ids)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels )
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    return train_dataloader, val_dataloader


#%%
def get_phobert_embedding():
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    phobert.eval()
    return tokenizer, phobert

class PhoEmbedding(object):
    def __init__(self):
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states = True)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.phobert.eval()

    def text_to_feature(self, max_seq_length, sents_a, *args, **kwargs):
        """ Converts a single `InputExample` into a single `InputFeatures`.
            :parameter example: A InputRecord instance represent a data instance
            :parameter label_list: List of possible labels for predicting
            :parameter max_seq_length: The maximum input sequence length for embedding
            :parameter tokenizer: A BERT-based tokenier to tokenize text
        """

        sents_a = ' '.join(vcn_word_segment(sents_a))
        # Text tokenization
        tokens_a = self.tokenizer.tokenize(sents_a)
        tokens_b = None
        if 'sents_b' in kwargs:
            sents_b = kwargs['sents_b']
            sents_b = ' '.join(vcn_word_segment(sents_b))
            tokens_b = self.tokenizer.tokenize(sents_b)

        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            """Truncates a sequence pair in place to the maximum length."""

            # This is a simple heuristic which will always truncate the longer sequence
            # one token at a time. This makes more sense than truncating an equal percent
            # of tokens from each, since if one sequence is very short then each token
            # that's truncated likely contains more information than a longer sequence.
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        # Truncate text if total length of combinec input > max sequence length for the model
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for <s>, </s>, </s> with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for <s> and </s> with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in RoBERTa is:
        # (a) For sequence pairs:
        #  tokens:   <s> is this jack ##son ##ville ? </s> no it is not . </s>
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   <s> the dog is hairy . </s>
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the </s> token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to <s>) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        tokens.append("<s>")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("</s>")
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
            tokens.append("</s>")

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        while len(input_ids) < max_seq_length:
            input_ids.append(1) # <pad> = 1
            attention_mask.append(0)
        
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length

        if 'label' in kwargs:
                    # Labels mapping
            label_map = {}
            for (i, label) in enumerate(kwargs['label_list']):
                label_map[label] = i
            label_id =  label_map[kwargs['label']]
        else:
            label_id = -1
        guid = kwargs['guid'] if 'guid' in kwargs else None,
        feature = RobertaInputFeatures(
            guid=guid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            label_id=label_id,
            is_real_example=True)
        return feature
    
    def feature_to_tensor(self, features, is_label):
        inputs = []
        if is_label:
            labels = []
        attention_masks  = []
        for feature in tqdm(features):
            inputs.append(feature.input_ids)
            if is_label:
                labels.append(feature.label_id)
            attention_masks.append(feature.attention_mask)
        inputs = torch.tensor(inputs)
        attention_masks = torch.tensor(attention_masks)
        if is_label:
            labels = torch.tensor(labels)
            return inputs, attention_masks, labels
        return inputs, attention_masks
    
    def convert_single_example(self, sample, label_list, max_seq_length, tokenizer):
        return self.text_to_feature(max_seq_length, sample.question, sents_b=sample.text, label=sample.label, label_list=label_list)

    def embvec(self, b_input_ids, b_attention_mask, mode='concat'):
        with torch.no_grad():
            outputs = self.phobert(b_input_ids, b_attention_mask)
            hidden_states = outputs[2] # shape [13 layers x batch_size x num_tokens x 768 hidden units]
        if mode == 'concat':
            return hidden_states, torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]), dim=-1)
