# -*- coding: utf-8 -*-
import pandas as pd
from fuzzywuzzy import fuzz
import gensim
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis
import cPickle
import sys


stop_words = stopwords.words('english')


def wmd(model, s1, s2):
    s1 = str(s1).lower().decode('utf-8').split()
    s2 = str(s2).lower().decode('utf-8').split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(model, s1, s2):
    s1 = str(s1).lower().decode('utf-8').split()
    s2 = str(s2).lower().decode('utf-8').split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if w not in stop_words]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def add_basic_feat(data):
    data['len_q1'] = data['question1'].apply(lambda x: len(str(x)))
    data['len_q2'] = data['question2'].apply(lambda x: len(str(x)))
    data['diff_len'] = data['len_q1'] - data['len_q2']
    data['len_char_q1'] = data['question1'].apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_char_q2'] = data['question2'].apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_word_q1'] = data['question1'].apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data['question2'].apply(lambda x: len(str(x).split()))
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)


def add_fuzz_feat(data):
    data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_wratio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


def add_word2vec_feat(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    data['wmd'] = data.apply(lambda x: wmd(model, x['question1'], x['question2']), axis=1)

    norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    norm_model.init_sims(replace=True)
    data['norm_wmd'] = data.apply(lambda x: norm_wmd(norm_model, x['question1'], x['question2']), axis=1)

    question1_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data['question1'].values)):
        question1_vectors[i, :] = sent2vec(q)

    question2_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data['question2'].values)):
        question2_vectors[i, :] = sent2vec(q)

    # cPickle.dump(question1_vectors, open('data/q1_w2v.pkl', 'wb'), -1)
    # cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)

    data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
    data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

    data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "usage:%s input_file output_file" % sys.argv[0]
        sys.exit(-1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = pd.read_csv(input_file)

    add_basic_feat(data)
    add_fuzz_feat(data)
    add_word2vec_feat(data)

    data.to_csv(output_file, index=False)
