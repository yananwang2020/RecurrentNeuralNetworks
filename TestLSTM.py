from RNN.LSTM import RNN_LSTM
from pathlib import Path
import numpy as np

sentence_file = Path('./Lib/stanfordSentimentTreebank/datasetSentences.txt')
lable_file = Path('./Lib/stanfordSentimentTreebank/sentiment_labels.txt')
param_file_path = Path('./Param/stanfordSentiment.pkl')

sentence_word_dict = {}
sentence_idx_dixt = {}
lable_dict = {}
word_set = set()

word_to_idx = {}
idx_to_word = {}

with sentence_file.open('r') as fr:
    sls = fr.readlines()

for sl in sls:
    k, v = sl.lower().split('\t')
    if(k.isdigit()):
        wl = v.replace('\n','').split(' ')
        sentence_word_dict[int(k)] = wl
        word_set.update(wl)

for k, v in enumerate(word_set)
    word_to_idx[v]=k
    idx_to_word[k]=v

for k, v in sentence_word_dict.items():
    idxlist = [ word_to_idx[word] for word in v]
    sentence_idx_dixt[k]=idxlist

with lable_file.open('r') as fr:
    lablelist = fr.readlines()

for lablestr in lablelist:
    k, v = lablestr.split('\t')
    if(k.isdigit()):
        lable_dict[int(k)] = v

