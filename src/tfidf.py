#%%
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import json
#%%

os.chdir("./Dataset")

#%%
train_df = pd.read_json('./train.json', orient='records')
with open('./test.json', 'r') as test_file:
    test_df = pd.json_normalize(json.load(test_file), 'paragraphs', meta=['__id__', 'question', 'title'], record_prefix='para_')
#%%
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("./VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
# rdrsegmenter = VnCoreNLP("/content/drive/My Drive/BERT/SA/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

text = "Đại học Công nghệ."

word_segmented_text = rdrsegmenter.tokenize(text) 
print(word_segmented_text)