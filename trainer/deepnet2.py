# -*- coding: utf-8 -*-
import os
import argparse
import ConfigParser
import cPickle as pickle
import pandas as pd
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
from tensorflow.python.lib.io import file_io


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gcp',
        help='program in local or gcp',
        action='store_true'
    )

    parser.add_argument(
        '--param_config',
        help='param_config_file',
        required=True
    )

    parser.add_argument(
        '--data_path',
        help='data path',
        required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
    )

    config = ConfigParser.ConfigParser()
    args = parser.parse_args()
    with file_io.FileIO(args.param_config, 'r') as f:
        config.readfp(f)

    nb_words = config.getint('common', 'nb_words')
    embedding_dim = config.getint('common', 'embedding_dim')
    max_sequence_length = config.getint('common', 'max_sequence_length')
    batch_size = config.getint('common', 'batch_size')
    nb_epoch = config.getint('common', 'nb_epoch')
    validation_split = config.getfloat('common', 'validation_split')

    model_file = config.get('common', 'model_file')
    submission_file = config.get('common', 'submission_file')
    history_file = config.get('common', 'history_file')

    with file_io.FileIO(args.data_path, 'r') as f:
        all_data = pickle.load(f)
    (train_q1, train_q2, test_q1, test_q2, embedding_matrix, labels, test_id) = all_data

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

    checkpoint = ModelCheckpoint(model_file, monitor='val_acc', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

    history = model.fit([train_q1, train_q2], y=labels, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=validation_split, shuffle=True, callbacks=[checkpoint, early_stopping])

    model.load_weights(model_file)

    pred = model.predict([test_q1, test_q2], batch_size=batch_size, verbose=1)

    sub = pd.DataFrame()
    sub['test_id'] = test_id
    sub['is_duplicate'] = pred
    sub.to_csv(submission_file, index=False)

    min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
    print 'min_val_loss', idx + 1, min_val_loss

    history_pd = pd.DataFrame({'epoch': [i + 1 for i in history.epoch],
                               'training': history.history['loss'],
                               'validation': history.history['val_loss']})
    history_pd.to_csv(history_file, index=False)

    if args.gcp:
        subprocess.check_call(['gsutil', '-m', 'cp', '-r', model_file, os.path.join(args.job_dir, model_file)])
        subprocess.check_call(['gsutil', '-m', 'cp', '-r', submission_file, os.path.join(args.job_dir, submission_file)])
        subprocess.check_call(['gsutil', '-m', 'cp', '-r', history_file, os.path.join(args.job_dir, history_file)])

    # plot
    # ax = history.ix[:, :].plot(x='epoch', figsize={5, 8}, grid=True)
    # ax.set_ylabel('loss')
    # ax.set_ylim([0.0, 1.0])
