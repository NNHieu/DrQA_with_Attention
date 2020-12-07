from readers.DrQA.data import Tokenizer
from fastai.data.all import *
from fastai.text.all import *
from fastai.torch_core import *


class SimpleTokenizer(Transform):
    def __init__(self, vocab):
        self.vocab = vocab
        
    def tok2id(self, tokens):
#         if(len(tokens) < 100):
#             print(tokens)
#             print([self.vocab[tok] for tok in tokens])
        return tensor([self.vocab[tok.lower()] for tok in tokens], dtype=torch.int64)
    
    def ids2toks(self, ids):
        return ' '.join([self.vocab[ind] for ind in ids.tolist()])
        
    def encodes(self, x:str):
        t = TensorText(self.tok2id(x.lower().split()))
#         print('.', end='')
        return t
    
    def decodes(self, x):
        return TitledStr(self.ids2toks(x))
    
def simple_tokenize(items):
    return (L(doc) for doc in map(lambda x: x.split(),items))

from transformers import AutoTokenizer
class TransformerTokenizer(Transform):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
     def encodes(self, x:str):
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))