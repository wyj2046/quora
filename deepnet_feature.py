# -*- coding: utf-8 -*-
import cPickle as pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing import text
from keras.preprocessing import sequence


MAX_NUM_WORDS = 200000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 25


TRAIN_FILE = 'input/train.csv'
TEST_FILE = 'input/test.csv'
GLOVE_FILE = 'data/glove.840B.300d.txt'


if __name__ == '__main__':
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    tk = text.Tokenizer(num_words=MAX_NUM_WORDS)
    tk.fit_on_texts(list(train_data.question1.values.astype(str)) +
                    list(train_data.question2.values.astype(str)) +
                    list(test_data.question1.values.astype(str)) +
                    list(train_data.question1.values.astype(str)))

    word_index = tk.word_index

    embeddings_index = {}
    f = open(GLOVE_FILE)
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
    f.close()

    nb_words = min(MAX_NUM_WORDS, len(word_index))

    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in tqdm(word_index.items()):
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    train_q1_word_sequence = tk.texts_to_sequences(train_data.question1.values.astype(str))
    train_q2_word_sequence = tk.texts_to_sequences(train_data.question2.values.astype(str))
    test_q1_word_sequence = tk.texts_to_sequences(test_data.question1.values.astype(str))
    test_q2_word_sequence = tk.texts_to_sequences(test_data.question2.values.astype(str))

    train_q1 = sequence.pad_sequences(train_q1_word_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    train_q2 = sequence.pad_sequences(train_q2_word_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    test_q1 = sequence.pad_sequences(test_q1_word_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    test_q2 = sequence.pad_sequences(test_q2_word_sequence, maxlen=MAX_SEQUENCE_LENGTH)

    labels = np.array(train_data.is_duplicate.values, dtype=int)

    test_id = np.array(test_data.test_id.values, dtype=int)

    all_data = (train_q1, train_q2, test_q1, test_q2, embedding_matrix, labels, test_id)
    pickle.dump(all_data, open('data/all_data.pickle', 'w'), True)

    print 'nb_words:%d' % nb_words
