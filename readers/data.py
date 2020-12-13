from typing import List

# from vncorenlp import VnCoreNLP
import string
from functools import reduce
import unicodedata
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import pytorch_lightning as pl

import numpy as np

class Tokenizer(object):
    def tex2toks(self, text: str) -> List[str]:
        pass

class ReaderVocab(object):
    def __init__(self, spec_tokens, UNK_index, PAD_index):
        self.spec_tokens = spec_tokens
        self.START = len(spec_tokens)
        self.tok2ind = {stoken: i for i, stoken in enumerate(self.spec_tokens)}
        self.ind2tok = {i: stoken for i, stoken in enumerate(self.spec_tokens)}
        self.UNK_index = UNK_index
        if UNK_index < 0:
            self.UNK = '<unk not set>'
        else:
            self.UNK = spec_tokens[UNK_index]
        self.PAD_index = PAD_index
    
    @classmethod
    def from_file(cls, file):
        rec = cls(['', ''], 0, 1)
        rec.tok2ind = {}
        rec.ind2tok = {}
        rec.load(file)
        return rec

    @staticmethod
    def normalize(token):
        # http://vietunicode.sourceforge.net/
        # FastText Vietnamese use NFC
        return unicodedata.normalize('NFC', token) 

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token
    
    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in self.spec_tokens]
        return tokens
    
    def save(self, file_path):
        with open(file_path, 'w') as f:
            for ind, tok in self.ind2tok.items():
                f.write(f'{ind} {tok}\n')
    
    def load(self, file_path):
        with open(file_path, 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                index, token = line.split()
                token = token.lower()
                index = int(index)
                self.tok2ind[token] = index
                self.ind2tok[index] = token

class W2VVocab(ReaderVocab):
    def __init__(self, spec_tokens, UNK_index, PAD_index, wv):
        super(W2VVocab, self).__init__(spec_tokens, UNK_index, PAD_index)
        for tok in wv.vocab:
            self.add(tok)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, pos_vocab, ner_vocab, max_seq_length, encodings, aug=None):
        self.encodings = encodings
        self.aug = aug
        self.len_enc = len(self.encodings['labels'])
        self.vocab = vocab
        self.ner_vocab =  ner_vocab
        self.pos_vocab = pos_vocab
        self.max_seq_length = max_seq_length
        if aug is not None:
            self.len_aug = len(self.aug['labels'])
            print('len aug', self.len_aug)
    
    def _single_toksnfeat2ids(self, text_toks:str, pos_toks:str, ner_toks:str):
        try:
            text_toks = text_toks.strip().lower().split(' ')
            pos_toks = pos_toks.strip().lower().split(' ')
            ner_toks = ner_toks.strip().lower().split(' ')
        except Exception:
            print('..........')
            print(text_toks, pos_toks, ner_toks)
        tok_ids = []
        pos_ids = []
        ner_ids = []
        for i in range(len(text_toks)):
            if i >= self.max_seq_length:
                break
            try:
                tok_ids.append(self.vocab[text_toks[i]])
                pos_ids.append(self.pos_vocab[pos_toks[i]])
                ner_ids.append(self.ner_vocab[ner_toks[i]])
            except:
                print('sda..........')
                
                print(text_toks, pos_toks, ner_toks)

        return tok_ids, pos_ids, ner_ids
    
    def idsitem(self, item):
        x1, pos1, ner1 = self._single_toksnfeat2ids(item['x1'], item['x1_f'][0], item['x1_f'][1])
        x2, pos2, ner2 = self._single_toksnfeat2ids(item['x2'], item['x2_f'][0], item['x2_f'][1])
        iitem = {'x1': x1, 'x2': x2, 'x1_f': [pos1, ner1], 'x2_f': [pos2, ner2]}        
        return iitem
    
    def _get_non_aug(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items() if key != 'labels'}
        item = self.idsitem(item)
        item['labels'] = int(self.encodings['labels'][idx])
        return item
    
    def _get_aug(self, idx):
        try:
            item = {key: val[idx] for key, val in self.aug.items() if key != 'labels'}
            item = self.idsitem(item)
            item['labels'] = int(self.aug['labels'][idx])
        except:
            idx = 0
            item = {key: val[idx] for key, val in self.aug.items() if key != 'labels'}
            item = self.idsitem(item)
            item['labels'] = int(self.aug['labels'][idx])
        return item

    def __getitem__(self, idx):
        if self.aug is not None and torch.rand((1, )) < 0.45:
            return self._get_aug(idx % self.len_aug)
        else:
            return self._get_non_aug(idx % self.len_enc)
        return item

    def __len__(self):
        if self.aug is None:
            return self.len_enc
        else:
            return self.len_enc + self.len_aug
            
    def lengths(self):
        ret = [(len(ex[0]), len(ex[1]))
                for ex in zip(self.encodings['x1'], self.encodings['x2'])]
        rm = []
        if self.aug is not None:
            for ex in zip(self.encodings['x1'], self.encodings['x2']):
                try:
                    ret.append((len(ex[0]), len(ex[1])))
                except:
                    continue
        return ret

def _pad(l, val, max_len):
    l += [val]*(max_len - len(l))
    return l
def _pad_group(g):
    max_len = max([len(x) for x in g[0]])
    mask = []
    for i in range(len(g[0])):
        mask.append([0]*len(g[0][i]) + [1]*(max_len - len(g[0][i])))
        _pad(g[0][i], 1, max_len)
        _pad(g[1][i][0], 1, max_len)
        _pad(g[1][i][1], 1, max_len)
#         print(g)
    return mask
def pad(batch):
#     print(batch)
    features = ['x1','x1_f', 'x2', 'x2_f', 'labels']
    d = {f:[] for f in features}
    for i in batch:
        for f in features:
            d[f].append(i[f])
    q = [d['x1'], d['x1_f']]
    p = [d['x2'], d['x2_f']]
    d['x1_mask'] = _pad_group(q)
    d['x2_mask'] = _pad_group(p)
    for k, v in d.items():
        try:
            d[k] = torch.tensor(v)
        except Exception:
            print(v)
    return d

class SortedBatchSampler(Sampler):

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array([(-l[0], -l[1], np.random.random()) for l in self.lengths],dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)])
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)


class GensimDataModule(pl.LightningDataModule):
    tok_field = 'toks'
    pos_field = 'pos'
    ner_field = 'ner'
    
    q_pref = 'q_'
    t_pref = 't_'
    num_labels = 2

    def __init__(
        self,
        vocab,
        pos_vocab,
        ner_vocab,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.vocab = vocab
        self.pos_vocab = pos_vocab
        self.ner_vocab = ner_vocab
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def filtnan(self, df):
        tmp = df.replace(np.nan, '', regex=True)
        return tmp[(tmp['t_toks'] != '') & (tmp['q_toks'] != '')]
    
    def setup(self, stage, train_df, valid_df, test_df=None, aug_df=None):
        if aug_df is not None:
            self.dataset = {
                "train": SimpleDataset(self.vocab, self.pos_vocab, self.ner_vocab, self.max_seq_length, self.convert_to_features(train_df), self.convert_to_features(aug_df)),
                "validation": SimpleDataset(self.vocab, self.pos_vocab, self.ner_vocab, self.max_seq_length, self.convert_to_features(valid_df)),
                }
        else:
            self.dataset = {
                "train": SimpleDataset(self.vocab, self.pos_vocab, self.ner_vocab, self.max_seq_length, self.convert_to_features(train_df)),
                "validation": SimpleDataset(self.vocab, self.pos_vocab, self.ner_vocab, self.max_seq_length, self.convert_to_features(valid_df)),
                }
        if test_df is not None:
            self.dataset['test'] = SimpleDataset(self.vocab, self.pos_vocab, self.ner_vocab, self.max_seq_length, self.convert_to_features(test_df))
        self.eval_splits = ["validation"]
    def prepare_data(self):
        pass
    def train_dataloader(self):
        sampler = SortedBatchSampler(
            self.dataset['train'].lengths(),
            self.train_batch_size,
            shuffle=False
        )
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, collate_fn=pad, sampler=sampler, num_workers=4, pin_memory=True)
    def val_dataloader(self):
        sampler = SortedBatchSampler(
            self.dataset['validation'].lengths(),
            self.eval_batch_size,
            shuffle=False
        )
        return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size, collate_fn=pad, sampler=sampler, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        if 'test' in self.dataset:
            return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size, collate_fn=pad,num_workers=4, pin_memory=True)
        return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size, collate_fn=pad,num_workers=4, pin_memory=True)

    def convert_to_features(self, example_batch, indices=None):
        features = {
            'x1': example_batch[self.q_pref + self.tok_field].tolist(),
            'x1_f': list(zip(example_batch[self.q_pref + self.pos_field], example_batch[self.q_pref + self.ner_field])),
            'x2': example_batch[self.t_pref + self.tok_field].tolist(),
            'x2_f': list(zip(example_batch[self.t_pref + self.pos_field], example_batch[self.t_pref + self.ner_field])),
            'labels': example_batch['label']
        }
        return features