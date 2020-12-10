# -*- coding: utf-8 -*-
import os
from gensim.models.fasttext import FastText
import re
from tqdm import tqdm
import time
import os
import pandas as pd
import string
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def continue_train(corpus_path, model, epochs):
    model.build_vocab(corpus_file=corpus_path, update=True)
    model.train(corpus_file=corpus_path, total_words=model.corpus_total_words, epochs=epochs)
    return model

def train(corpus_path, epochs, size, window, min_count, workers=2, sg=1, max_vocab_size=10000):
    model = FastText(size=size, window=window, min_count=min_count, workers=workers, sg=sg, max_vocab_size=max_vocab_size)
    model.build_vocab(corpus_file=corpus_path, update=False)
    model.train(corpus_file=corpus_path, total_words=model.corpus_total_words, epochs=epochs)
    return model

def reduced_dim(words, wv):
    words_np = []
    words_label = []

    for word in words:
        if word in wv.vocab.keys():
            print(word)
            words_np.append(wv[word])
            words_label.append(word)

    pca = PCA(n_components=2)
    pca.fit(words_np)
    return words_label, pca.transform(words_np)

def visualize(words_label, reduced):
    fig, ax = plt.subplots()
    for index, vec in enumerate(reduced):
        x, y = vec[0], vec[1]
        ax.scatter(x, y)
        ax.annotate(words_label[index], xy=(x, y))
    plt.show()
    return