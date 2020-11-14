import pandas as pd
import json
from tqdm import tqdm


class ZaloTextPreprocess(object):

    def __init__(self):
        pass

    def load_data(self, dataset_path):
        self.train_df = pd.read_json(f'{dataset_path}/train.json', orient='records')
        with open(f'{dataset_path}/test.json', 'r') as test_file:
            self.test_df = pd.json_normalize(json.load(test_file), 'paragraphs', meta=['__id__', 'question', 'title'], record_prefix='para_')
            self.test_df = self.test_df.rename(columns={'__id__': 'id', 'para_text': 'text'})
            self.test_df['id'] = self.test_df['id'] + '_' + self.test_df['para_id']
            self.test_df = self.test_df.drop(columns='para_id')
            self.test_df = self.test_df[['id', 'question', 'title', 'text']]
    
    def convert_to_features(self, preprocessor, tokenizer, *args, **kwargs):
        features = []
        for index, row in tqdm(self.train_df.iterrows(), total=self.train_df.shape[0]):
            features.append(preprocessor.convert_single_example(row, [0, 1], kwargs['max_seq_length'], tokenizer))
        return features

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
