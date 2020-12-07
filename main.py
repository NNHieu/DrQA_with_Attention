#%%
from functools import partial
import os
from posix import listdir
from textprocessor.normalization import clean_text, remove_punctuation, remove_stopword
from textprocessor import normalization, text_transform, statistic
import re
import pandas as pd
#%%
text_transform.process_corpus('dataset/corpus/viwik19/lower/viwik19_aa', 'dataset/corpus/viwik19/lower2/viwik19_aa', [normalization.remove_punctuation])
# %%
fs = [ f for f in os.listdir('dataset/corpus/viwik19/lower')]
corpus_path = 'dataset/corpus/'
stop = {}
stopwords = set(pd.read_csv('./textprocessor/stopwords.csv')['stopwords'])
for f in fs:
    text_transform.process_corpus('dataset/corpus/viwik19/lower/'+f, 'dataset/corpus/viwik19/lower2/'+f, [normalization.remove_punctuation, partial(remove_stopword, stopwords, join=True), clean_text])
    # c, total, s = statistic.token_statistic('dataset/corpus/viwik19/lower2/'+f, 'dataset/corpus/viwik19/lower2/'+f +'.stat', per_sent=True)
    # s = []
    # # with open('dataset/corpus/viwik19/lower2/' + f + '.stat', 'r') as stat:
    # #     line = stat.readline()
    # #     while line:
    # #         line = line.split(' ')
    # #         print(line)
    # #         if(len(line) < 2):
    # #             continue
    # #         s.append((line[0], float(line[1])))
    # #         line = stat.readline()
    # l = len(s)
    # s = s[:int(l*0.6)]
    # for k, v in s:
    #     if k in stop:
    #         stop[k] += v
    #     else:
    #         stop[k] = v    
# %%

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
# %%

# %%
