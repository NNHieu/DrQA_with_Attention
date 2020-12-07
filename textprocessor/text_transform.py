#%%
import enum
from os import name
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
import itertools 
import tqdm
import json
from textprocessor.normalization import *
#%%
def load_train_data(dataset_path, train_name='train.json'):
    return pd.read_json(f'{dataset_path}/{train_name}', orient='records')

def load_test_data(dataset_path, test_name='test.json'):
    with open(f'{dataset_path}/{test_name}', 'r') as test_file:
        test_df = pd.json_normalize(json.load(test_file), 'paragraphs', meta=['__id__', 'question', 'title'], record_prefix='para_')
        test_df = test_df.rename(columns={'__id__': 'id', 'para_text': 'text'})
        test_df['id'] = test_df['id'] + '_' + test_df['para_id']
        test_df = test_df.drop(columns='para_id')
        test_df = test_df[['id', 'question', 'title', 'text']]
    return test_df

def _split_ds(df, sz):
    assert reduce(lambda x, y: x + y, sz) == 1
    if len(sz) == 1:
        return [df]
    con, ret = train_test_split(df, test_size=sz[0])
    su = sum(sz[1:])
    sz = list(map(lambda x: x/su, sz[1:]))
    return [ret, *_split_ds(con, sz)]

def split_ds(df, sz, label_col='label'):
    pdf = df[df['label']]
    ndf = df[df['label'] == False]
    pdfs = _split_ds(pdf, sz)
    ndfs = _split_ds(ndf, sz)
    dfs = []
    for i in range(len(pdfs)):
        dfs.append(pd.concat([pdfs[i], ndfs[i]]))
    return dfs

def df2corpus(filename, df, cols=['question', 'text'], pros=None):
    with open(filename, 'w') as f:
        for i, r in df.iterrows():
            for col in cols:
                if pros:
                    f.write(apply_textprocess(r[col], pros) + '\n')
                else:
                    f.write(r[col] + '\n')


def process_corpus(filename, outfile, pros):
    with open(filename, 'r') as infi:
        with open(outfile, 'w') as oufi:
            line = infi.readline()
            while(line):
                oufi.write(apply_textprocess(line, pros) + '\n')
                line = infi.readline()

def split_dataset(df, sz):
    clean_df(df)
    dfs = split_ds(df, sz)
    dfs[0].to_csv('train_set.csv')
    dfs[1].to_csv('valid_set.csv')
    dfs[2].to_csv('test_set.csv')

def _vncore_feature(text, pre, filters=None):
    anno = VncorenlpTokenizer.getInstance(True, True).annotate(text)
    sample = {f'{pre}_toks': '', f'{pre}_pos':'' ,f'{pre}_ner': '',}
    for sent in anno['sentences']:
        for tok in sent:
            if filters is not None and all(f(tok['form']) for f in filters):
                continue
            sample[f'{pre}_toks'] += tok['form'] + ' '
            sample[f'{pre}_pos'] += tok['posTag'] + ' '
            sample[f'{pre}_ner'] += tok['nerLabel'] + ' '
    return sample
                   
def _vncore_feature_row(row, cols={'question':'q', 'text':'t'}, filters=None):
    val = row
    for k, v in cols.items():
        val = dict(val,**_vncore_feature(row[k], v, filters))
    return val

def create_feature_df(df, cols={'question':'q', 'text':'t'}, drop=False, filters=None):
    sample = _vncore_feature_row(df.iloc[0],  cols=cols)
    feat_df = pd.DataFrame(columns=[*sample.keys()])
    for i, r in df.iterrows():
        feat_df = feat_df.append(_vncore_feature_row(r, cols=cols, filters=filters), ignore_index=True)
    return feat_df.drop(columns=list(cols.keys()))

def _filter_row(row, filters, group, reverse_check=False):
    
    for tok_col, tok_feats in group.items():
        str_tok_col = ''
        str_tok_feats = ['' for i in tok_feats]
        toks = row[tok_col].split(' ')
        token_feats = [row[tok_feat].split(' ') for tok_feat in tok_feats]
        for j, tok in enumerate(toks):
            check = all((f(tok) and not reverse_check) or (not f(tok) and reverse_check) for f in filters)
            if check :
                str_tok_col += tok + ' '
                for i in range(len(tok_feats)):
                    str_tok_feats[i] += token_feats[i][j] + ' '
        row[tok_col] = str_tok_col.strip()
        for i, tf in enumerate(tok_feats):
            row[tf] = str_tok_feats[i].strip()
        assert all(len(str_tok_col.split(' ')) == len(sf.split(' ')) for sf in str_tok_feats)
    return row

def filter_toknfeat_df(df, filters, group={'q_toks': ['q_pos', 'q_ner'], 't_toks': ['t_pos', 't_ner']}, reverse_check=False):
    res = pd.DataFrame(columns=df.columns)
    for i, r in df.iterrows():
        res = res.append(_filter_row(r, filters, group, reverse_check), ignore_index=True)
        # print(res)
    return res
# %%
