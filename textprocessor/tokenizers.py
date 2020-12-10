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

class DrQABatchTransform:
    tok_field = 'toks'
    pos_field = 'pos'
    ner_field = 'ner'
    
    q_pref = 'q_'
    t_pref = 't_'
    
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
    
     def _single_toksnfeat2ids(self, text_toks:str, pos_toks:str, ner_toks:str):
        text_toks = text_toks.strip().lower().split(' ')
        pos_toks = pos_toks.strip().lower().split(' ')
        ner_toks = ner_toks.strip().lower().split(' ')
        tok_ids = []
        pos_ids = []
        ner_ids = []
        for i in range(len(text_toks)):
            if i >= self.max_seq_length:
                break
            tok_ids.append(self.vocab[text_toks[i]])
            pos_ids.append(self.pos_vocab[pos_toks[i]])
            ner_ids.append(self.ner_vocab[ner_toks[i]])
        return tok_ids, pos_ids, ner_ids
    
    def pad(self, l, val, max_len):
        l += [val]*(max_len - len(l))
        return l
    
    def toksnfeat2ids(self, tok, pos, ner, padding=True):
        assert len(tok) == len(pos) and len(tok) == len(ner)
        longest = 0
        batch_tok_ids = []
        batch_pos_ids = []
        batch_ner_ids = []
        batch_mask = []
        for i in range(len(tok)):
            try:
                if len(tok[i]) <= 0:
                    continue

                tok_ids, pos_ids, ner_ids = self._single_toksnfeat2ids(tok[i], pos[i], ner[i])
                longest = max(longest, len(tok_ids))
                batch_tok_ids.append(tok_ids)
                batch_pos_ids.append(pos_ids)
                batch_ner_ids.append(ner_ids)
                batch_mask.append([0]*len(tok_ids))
            except TypeError:
#                 print('Skip ', tok[i])
                continue
#         print(pos[0:5])
#         print(batch_pos_ids[0:5])
#         print(ner[0:5])
        
        for i in range(len(batch_tok_ids)):
            self.pad(batch_mask[i], 1, longest)
            self.pad(batch_tok_ids[i], self.vocab['<pad>'], longest)
            self.pad(batch_pos_ids[i], self.pos_vocab['<pad>'], longest)
            self.pad(batch_ner_ids[i], self.ner_vocab['<pad>'], longest)
        
        return batch_tok_ids, batch_pos_ids, batch_ner_ids, batch_mask
    
    def __call__(self, batch):
        qids= self.toksnfeat2ids(example_batch[self.q_pref + self.tok_field], 
                                     example_batch[self.q_pref + self.pos_field], 
                                     example_batch[self.q_pref + self.ner_field])
        pids= self.toksnfeat2ids(example_batch[self.t_pref + self.tok_field], 
                                     example_batch[self.t_pref + self.pos_field], 
                                     example_batch[self.t_pref + self.ner_field])
        
        features = {
            'x1': qids[0],
            'x1_f': list(zip(qids[1], qids[2])),
            'x1_mask': qids[3],
            'x2': pids[0],
            'x2_f': list(zip(pids[1], pids[2])),
            'x2_mask': pids[3],
            'labels': example_batch['label']
        }
        return features