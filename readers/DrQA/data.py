from typing import List

# from vncorenlp import VnCoreNLP
import string
from functools import reduce
import unicodedata
import pickle

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
