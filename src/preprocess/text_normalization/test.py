#%%
import re
from gensim.summarization.textcleaner import get_sentences, clean_text_by_sentences
#%%
def sentence_segment(text):
    sents = [sent for sent in get_sentences(text)]
    return sents
#%%
def word_segment(sent, tokenize):
    sent = tokenize(sent)
    return sent
#%%
def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()
# %%
# %%
import sentencepiece as spm

def create_sentencepiece_model(corpus, model_prefix, vocab_size, *args, **kargs):
    config = f'--input={corpus} --model_prefix={model_prefix} --vocab_size={vocab_size}'
    for k,v in kargs.items():
        config += f' --{k}={v}'
    print(config)
    spm.SentencePieceTrainer.train(config)
    
# %%
create_sentencepiece_model('Dataset/viwik19.txt', 'viwik', 64003, user_defined_symbols='<sep>,<cls>')
# %%
create_sentencepiece_model('Dataset/viwik19/dataset/viwik19_aa', 't', 1000,  user_defined_symbols='<sep>,<cls>')
# %%
sp_user = spm.SentencePieceProcessor()
sp_user.load('Dataset/viwik19/viwik_60k_multi_word.model')
# %%
print(sp_user.encode_as_pieces())
# %%
from vncorenlp import VnCoreNLP
# %%
annotator = VnCoreNLP("models/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 
# %%
# Tokenizer cua phobert
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#%%
test_str = 'Kể lại chuyện, ông Nguyễn Văn Hoan, Chủ tịch Ủy ban nhân dân xã Tân Ninh, huyện Quảng Ninh, Quảng Bình "nói thật" với tôi: "địa phương chỉ có hai chiếc đò cole cũ". Trong đêm lụt tháng 10 vừa qua, chính quyền xã dùng đò đi cứu dân. Một chiếc không bật nổi sóng để đi. Một chiếc bị chìm, ông Hoan và bốn người rơi xuống nước.'
#%%
tokenizer.tokenize(test_str)
# %%
annotator.tokenize(test_str)
# %%
