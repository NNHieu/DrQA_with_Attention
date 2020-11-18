import pandas as pd
import json
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler

from vncorenlp import VnCoreNLP

class VncorenlpTokenizer(object):
    __instance = None
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if VncorenlpTokenizer.__instance == None:
            VncorenlpTokenizer()
        return VncorenlpTokenizer.__instance
    def __init__(self):
        """ Virtually private constructor. """
        if VncorenlpTokenizer.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            VncorenlpTokenizer.__instance = VnCoreNLP("models/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

    @staticmethod
    def word_segment(text):
        # To perform word (and sentence) segmentation
        sentences = VncorenlpTokenizer.getInstance().tokenize(text) 
        return [" ".join(sentence) for sentence in sentences]

class ZaloTextPreprocess(object):

    def __init__(self):
        self.features = None
        pass

    def load_data(self, dataset_path):
        self.train_df = pd.read_json(f'{dataset_path}/train.json', orient='records')
        with open(f'{dataset_path}/test.json', 'r') as test_file:
            self.test_df = pd.json_normalize(json.load(test_file), 'paragraphs', meta=['__id__', 'question', 'title'], record_prefix='para_')
            self.test_df = self.test_df.rename(columns={'__id__': 'id', 'para_text': 'text'})
            self.test_df['id'] = self.test_df['id'] + '_' + self.test_df['para_id']
            self.test_df = self.test_df.drop(columns='para_id')
            self.test_df = self.test_df[['id', 'question', 'title', 'text']]
    
    def convert_to_features(self, text_to_feature, max_seq_length, *args, **kwargs):
        self.features = []
        for index, row in tqdm(self.train_df.iterrows(), total=self.train_df.shape[0]):
            self.features.append(text_to_feature(max_seq_length, row.question, sents_b=row.text, label=row.label, label_list=[0, 1]))
        return self.features

    def convert_to_dataloader(self, preprocessor, max_seq_length, batch_size=16, dev_size=0.2, force_features=False):
        if self.features is None or force_features:
            self.convert_to_features(preprocessor.text_to_feature, max_seq_length)
        tensors = preprocessor.features_to_tensor(self.features, True)
        return preprocessor.tensor_to_dataloader(tensors, batch_size=batch_size, dev_size=dev_size)

class BertInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

class RobertaInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 attention_masks,
                 label_id,
                 is_real_example=True):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label_id = label_id
        self.is_real_example = is_real_example

class PhobertTextProcessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def text_to_tokens(self, max_seq_length, sents_a, *args, **kwargs):
        sents_a = ' '.join(VncorenlpTokenizer.word_segment(sents_a))
        # Text tokenization
        tokens_a = self.tokenizer.tokenize(sents_a)
        tokens_b = None
        if 'sents_b' in kwargs:
            sents_b = kwargs['sents_b']
            sents_b = ' '.join(VncorenlpTokenizer.word_segment(sents_b))
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
        return tokens

    def text_to_feature(self, max_seq_length, sents_a, *args, **kwargs):
        """ Converts a single `InputExample` into a single `InputFeatures`."""

        tokens = self.text_to_tokens(max_seq_length, sents_a, args, kwargs)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_masks = [1] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        while len(input_ids) < max_seq_length:
            input_ids.append(1) # <pad> = 1
            attention_masks.append(0)
        
        assert len(input_ids) == max_seq_length
        assert len(attention_masks) == max_seq_length

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
            attention_masks=attention_masks,
            label_id=label_id,
            is_real_example=True)
        return feature
    
    def features_to_tensor(self, features, is_label):
        inputs = []
        if is_label:
            labels = []
        attention_masks  = []
        for feature in tqdm(features):
            inputs.append(feature.input_ids)
            if is_label:
                labels.append(feature.label_id)
            attention_masks.append(feature.attention_masks)
        inputs = torch.tensor(inputs)
        attention_masks = torch.tensor(attention_masks)
        if is_label:
            labels = torch.tensor(labels)
            return inputs, attention_masks, labels
        return inputs, attention_masks
    
    def tensor_to_dataloader(self, tensors, batch_size=16, dev_size=0.2, shuffle_dataset=True):
        input_ids, attention_masks = tensors[:2]
        if len(tensors) == 3:
            labels = tensors[2]
        else:
            labels = None
        
        assert input_ids.size(0) == attention_masks.size(0)
        assert labels is None or attention_masks.size(0) == labels.size(0)
        
        dataset_size = input_ids.size(0)
        print(dataset_size)
        print(dev_size)

        indices = list(range(dataset_size))
        split = int(np.floor(dev_size * dataset_size))
        if shuffle_dataset :
            random_seed = 42
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        if labels is not None:
            dataset = TensorDataset(input_ids, attention_masks, labels)
        else:
            dataset = TensorDataset(input_ids, attention_masks)
        
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
        validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
        return train_loader, validation_loader

class PaddingInputExample(object):
    """ Fake example so the num input examples is a multiple of the batch size.

        When running eval/predict on the TPU, we need to pad the number of examples
        to be a multiple of the batch size, because the TPU requires a fixed batch
        size. The alternative is to drop the last batch, which is bad because it means
        the entire output data won't be generated.

        We use this class instead of `None` because treating `None` as padding
        battches could cause silent errors.
    """

class BERTPreprocessor(object):
    def convert_single_example(self, sample, label_list, max_seq_length, bert_base_tokenizer):
        """ Converts a single `InputExample` into a single `BertInputFeatures`.
            :parameter example: A InputRecord instance represent a data instance
            :parameter label_list: List of possible labels for predicting
            :parameter max_seq_length: The maximum input sequence length for embedding
            :parameter tokenizer: A BERT-based tokenier to tokenize text
        """

        # Return dummy features if fake example (for batch padding purpose)
        if isinstance(sample, PaddingInputExample):
            return BertInputFeatures(
                guid="",
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label_id=0,
                is_real_example=False)

        # Labels mapping
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        # Text tokenization
        tokens_a = bert_base_tokenizer.tokenize(sample.question)
        tokens_b = None
        if sample.text:
            tokens_b = bert_base_tokenizer.tokenize(sample.text)

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
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = bert_base_tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[sample.label] if sample.label is not None else -1

        feature = BertInputFeatures(
            guid=sample.id,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature
