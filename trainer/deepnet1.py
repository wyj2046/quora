# -*- coding: utf-8 -*-
import os
import argparse
import ConfigParser
import cPickle as pickle
import pandas as pd
import subprocess
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers.advanced_activations import PReLU
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

    dropout = config.getfloat('common', 'dropout')
    filter_length = config.getint('common', 'filter_length')
    nb_filter = config.getint('common', 'nb_filter')
    pool_length = config.getint('common', 'pool_length')

    model_file = config.get('common', 'model_file')
    submission_file = config.get('common', 'submission_file')
    history_file = config.get('common', 'history_file')

    with file_io.FileIO(args.data_path, 'r') as f:
        all_data = pickle.load(f)
    (train_q1, train_q2, test_q1, test_q2, embedding_matrix, labels, test_id) = all_data

    model1 = Sequential()
    model1.add(Embedding(nb_words + 1,
                         embedding_dim,
                         weights=[embedding_matrix],
                         input_length=max_sequence_length,
                         trainable=False))

    model1.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
    model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(embedding_dim,)))

    model2 = Sequential()
    model2.add(Embedding(nb_words + 1,
                         embedding_dim,
                         weights=[embedding_matrix],
                         input_length=max_sequence_length,
                         trainable=False))

    model2.add(TimeDistributed(Dense(embedding_dim, activation='relu')))
    model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(embedding_dim,)))

    model3 = Sequential()
    model3.add(Embedding(nb_words + 1,
                         embedding_dim,
                         weights=[embedding_matrix],
                         input_length=max_sequence_length,
                         trainable=False))
    model3.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model3.add(Dropout(dropout))

    model3.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))

    model3.add(GlobalMaxPooling1D())
    model3.add(Dropout(dropout))

    model3.add(Dense(embedding_dim))
    model3.add(Dropout(dropout))
    model3.add(BatchNormalization())

    model4 = Sequential()
    model4.add(Embedding(nb_words + 1,
                         embedding_dim,
                         weights=[embedding_matrix],
                         input_length=max_sequence_length,
                         trainable=False))
    model4.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model4.add(Dropout(dropout))

    model4.add(Convolution1D(nb_filter=nb_filter,
                             filter_length=filter_length,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))

    model4.add(GlobalMaxPooling1D())
    model4.add(Dropout(dropout))

    model4.add(Dense(embedding_dim))
    model4.add(Dropout(dropout))
    model4.add(BatchNormalization())
    model5 = Sequential()
    model5.add(Embedding(nb_words + 1, embedding_dim, input_length=max_sequence_length, dropout=dropout))
    model5.add(LSTM(embedding_dim, dropout_W=dropout, dropout_U=dropout))

    model6 = Sequential()
    model6.add(Embedding(nb_words + 1, embedding_dim, input_length=max_sequence_length, dropout=dropout))
    model6.add(LSTM(embedding_dim, dropout_W=dropout, dropout_U=dropout))

    merged_model = Sequential()
    merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))
    merged_model.add(BatchNormalization())

    for _ in range(5):
        merged_model.add(Dense(embedding_dim))
        merged_model.add(PReLU())
        merged_model.add(Dropout(dropout))
        merged_model.add(BatchNormalization())

    merged_model.add(Dense(1))
    merged_model.add(Activation('sigmoid'))

    merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

    history = merged_model.fit([train_q1, train_q2, train_q1, train_q2, train_q1, train_q2], y=labels, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=validation_split, shuffle=True, callbacks=[checkpoint, early_stopping])

    merged_model.load_weights(model_file)

    pred = merged_model.predict([test_q1, test_q2, test_q1, test_q2, test_q1, test_q2], batch_size=batch_size, verbose=1)

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
