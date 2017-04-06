# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers import TimeDistributed
from keras.layers import Lambda
from keras import backend as K
from keras.layers import Merge
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


MAX_NUM_WORDS = 200000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 25
BATCH_SIZE = 384
NB_EPOCH = 25
VALIDATION_SPLIT = 0.1


if __name__ == '__main__':
    # python deepnet2.py input/train.csv input/test.csv model/deepnet2.h5 submission/deepnet2.csv data/glove.840B.300d.txt plot/history2.csv
    if len(sys.argv) != 7:
        print 'usage:%s train test model submission glove'
        sys.exit(-1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    submission_file = sys.argv[4]
    glove_file = sys.argv[5]
    history_file = sys.argv[6]

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    train_y = train_data.is_duplicate.values

    tk = text.Tokenizer(num_words=MAX_NUM_WORDS)
    tk.fit_on_texts(list(train_data.question1.values.astype(str)) +
                    list(train_data.question2.values.astype(str)) +
                    list(test_data.question1.values.astype(str)) +
                    list(train_data.question1.values.astype(str)))

    word_index = tk.word_index

    embeddings_index = {}
    f = open(glove_file)
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

    Q1 = Sequential()
    Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))

    Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))

    Q2 = Sequential()
    Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))

    model = Sequential()
    model.add(Merge([Q1, Q2], mode='concat'))
    model.add(BatchNormalization())

    for _ in range(4):
        model.add(Dense(200, activation='relu'))
        model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

    history = model.fit([train_q1, train_q2], y=train_y, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=1, validation_split=VALIDATION_SPLIT, shuffle=True, callbacks=[checkpoint, early_stopping])

    model.load_weights(model_file)

    pred = model.predict([test_q1, test_q2], batch_size=BATCH_SIZE, verbose=1)

    sub = pd.DataFrame()
    sub['test_id'] = test_data['test_id'].values
    sub['is_duplicate'] = pred
    sub.to_csv(submission_file, index=False)

    min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
    print idx + 1, min_val_loss

    history_pd = pd.DataFrame({'epoch': [i + 1 for i in history.epoch],
                               'training': history.history['loss'],
                               'validation': history.history['val_loss']})
    history_pd.to_csv(history_file, index=False)
    # plot
    # ax = history.ix[:, :].plot(x='epoch', figsize={5, 8}, grid=True)
    # ax.set_ylabel('loss')
    # ax.set_ylim([0.0, 1.0])
