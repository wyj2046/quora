# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import subprocess
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_q1',
        help='train_q1 data file',
        required=True
    )
    parser.add_argument(
        '--train_q2',
        help='train_q2 data file',
        required=True
    )
    parser.add_argument(
        '--test_q1',
        help='test_q1 data file',
        required=True
    )
    parser.add_argument(
        '--test_q2',
        help='test_q2 data file',
        required=True
    )
    parser.add_argument(
        '--labels',
        help='labels data file',
        required=True
    )
    parser.add_argument(
        '--test_id',
        help='test_id data file',
        required=True
    )
    parser.add_argument(
        '--embedding_matrix',
        help='embedding_matrix data file',
        required=True
    )
    parser.add_argument(
        '--model_file',
        help='model_file',
        required=True
    )
    parser.add_argument(
        '--submission_file',
        help='submission_file',
        required=True
    )
    parser.add_argument(
        '--history_file',
        help='history_file',
        required=True
    )
    parser.add_argument(
        '--nb_words',
        help='nb_words',
        type=int,
        required=True
    )
    parser.add_argument(
        '--embedding_dim',
        help='embedding_dim',
        type=int,
        default=300
    )
    parser.add_argument(
        '--max_sequence_length',
        help='max_sequence_length',
        type=int,
        default=25
    )
    parser.add_argument(
        '--batch_size',
        help='batch_size',
        type=int,
        default=384
    )
    parser.add_argument(
        '--nb_epoch',
        help='nb_epoch',
        type=int,
        default=25
    )
    parser.add_argument(
        '--validation_split',
        help='validation_split',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    args = parser.parse_args()

    subprocess.check_call(['gsutil', '-m', 'cp', '-r', os.path.join('gs://kaggle-quora/data/', args.train_q1), '/tmp'])
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', os.path.join('gs://kaggle-quora/data/', args.train_q2), '/tmp'])
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', os.path.join('gs://kaggle-quora/data/', args.test_q1), '/tmp'])
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', os.path.join('gs://kaggle-quora/data/', args.test_q2), '/tmp'])
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', os.path.join('gs://kaggle-quora/data/', args.labels), '/tmp'])
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', os.path.join('gs://kaggle-quora/data/', args.test_id), '/tmp'])
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', os.path.join('gs://kaggle-quora/data/', args.embedding_matrix), '/tmp'])

    train_q1 = np.load(open(os.path.join('/tmp', args.train_q1), 'rb'))
    train_q2 = np.load(open(os.path.join('/tmp', args.train_q2), 'rb'))
    test_q1 = np.load(open(os.path.join('/tmp', args.test_q1), 'rb'))
    test_q2 = np.load(open(os.path.join('/tmp', args.test_q2), 'rb'))

    labels = np.load(open(os.path.join('/tmp', args.labels), 'rb'))
    train_y = labels

    test_id = np.load(open(os.path.join('/tmp', args.test_id), 'rb'))

    embedding_matrix = np.load(open(os.path.join('/tmp', args.embedding_matrix), 'rb'))

    nb_words = args.nb_words
    embedding_dim = args.embedding_dim
    max_sequence_length = args.max_sequence_length
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    validation_split = args.validation_split

    Q1 = Sequential()
    Q1.add(Embedding(nb_words + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))

    Q1.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
    Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))

    Q2 = Sequential()
    Q2.add(Embedding(nb_words + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
    Q2.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
    Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))

    model = Sequential()
    model.add(Merge([Q1, Q2], mode='concat'))
    model.add(BatchNormalization())

    for _ in range(4):
        model.add(Dense(200, activation='relu'))
        model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(args.model_file, monitor='val_acc', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

    history = model.fit([train_q1, train_q2], y=train_y, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=validation_split, shuffle=True, callbacks=[checkpoint, early_stopping])

    subprocess.check_call(['gsutil', '-m', 'cp', '-r', args.model_file, os.path.join('gs://kaggle-quora/model/', os.path.basename(args.model_file))])

    model.load_weights(args.model_file)

    pred = model.predict([test_q1, test_q2], batch_size=batch_size, verbose=1)

    sub = pd.DataFrame()
    sub['test_id'] = test_id
    sub['is_duplicate'] = pred
    sub.to_csv(args.submission_file, index=False)
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', args.submission_file, os.path.join('gs://kaggle-quora/submission/', os.path.basename(args.submission_file))])

    min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
    print idx + 1, min_val_loss

    history_pd = pd.DataFrame({'epoch': [i + 1 for i in history.epoch],
                               'training': history.history['loss'],
                               'validation': history.history['val_loss']})
    history_pd.to_csv(args.history_file, index=False)
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', args.history_file, os.path.join('gs://kaggle-quora/plot/', os.path.basename(args.history_file))])
    # plot
    # ax = history.ix[:, :].plot(x='epoch', figsize={5, 8}, grid=True)
    # ax.set_ylabel('loss')
    # ax.set_ylim([0.0, 1.0])
