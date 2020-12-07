#%%
import re
from vncorenlp import VnCoreNLP
import string
from functools import partial

#%%
def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def clean_df(df, cols=['question', 'text']):
    for col in cols:
        df[col] = list(map(clean_text, df[col]))
    return df


def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents


def tokenize(text, join=False, sent_sep=' '):
    toks = VncorenlpTokenizer.getInstance().tokenize(text)
    if join:
        toks = sent_sep.join([' '.join(sent) for sent in toks])
    return toks

def tokenize_df(df, cols=['question', 'text'], sent_sep=' '):
    for col in cols:
        df[col] = list(map( partial(tokenize, join=True, sent_sep=sent_sep), df[col]))
    return df

class _NormalizeMeta(object):
    listpunctuation = string.punctuation.replace('_', '')
    stopwords = []
    @classmethod
    def set_stopwords(cls, stopwords):
        cls.stopwords = stopwords

def is_punctuation(tok):
    return tok in _NormalizeMeta.listpunctuation

def is_stopwords(tok):
    return tok in _NormalizeMeta.stopwords

def remove_punctuation(text):
    listpunctuation = _NormalizeMeta.listpunctuation
    listpunctuation.replace('.', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text

def remove_punctuation_df(df, cols=['question', 'text']):
    for col in cols:
        df[col] = list(map(remove_punctuation, df[col]))
    return df

def remove_stopword(stopwords, text, join=False):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in stopwords:
            pre_text.append(word)
    text = pre_text
    if join:
        text = ' '.join(text)
    return text
def remove_stopword_df(df, stopwords, cols=['question', 'text']):
    for col in cols:
        df[col] = list(map(partial(remove_stopword, stopwords=stopwords), df[col]))
    return df

def apply_textprocess(text, processes):
    processed_text = text
    for p in processes:
        processed_text = p(processed_text)
    return processed_text

class VncorenlpTokenizer(object):
    __instance = None
    ner = False
    pos = False
    @staticmethod 
    def getInstance(ner=False, pos=False):
        """ Static access method. """
        if VncorenlpTokenizer.__instance == None or (VncorenlpTokenizer.ner != ner or VncorenlpTokenizer.pos != pos):
            VncorenlpTokenizer(ner, pos)
        return VncorenlpTokenizer.__instance

    def __init__(self, ner=False, pos=False):
        if self.__instance is not None and not (VncorenlpTokenizer.ner != ner or VncorenlpTokenizer.pos != pos):
            raise Exception
        VncorenlpTokenizer.ner = ner
        VncorenlpTokenizer.pos = pos 
        mode = 'wseg'
        max_heap_size = '-Xmx500m'
        
        if ner: 
            mode += ',ner'
            max_heap_size = '-Xmx2g'
        if pos: 
            mode += ',pos'
            max_heap_size = '-Xmx2g'
        VncorenlpTokenizer.__instance = VnCoreNLP("textprocessor/vncorenlp/VnCoreNLP-1.1.1.jar", annotators=mode, max_heap_size=max_heap_size)
        
    @staticmethod
    def word_segment(text):
        # To perform word (and sentence) segmentation
        sentences = VncorenlpTokenizer.getInstance().tokenize(text) 
        return [" ".join(sentence) for sentence in sentences]