# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split


seed = 2017


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "usage:%s train_data test_data submission model" % sys.argv[0]
        sys.exit(-1)
    train_data_file = sys.argv[1]
    test_data_file = sys.argv[2]
    submission_file = sys.argv[3]
    model_file = sys.argv[4]

    train_data = pd.read_csv(train_data_file)
    train_data.drop(['id', 'qid1', 'qid2', 'question1', 'question2'], axis=1, inplace=True)
    y_train = train_data['is_duplicate'].values
    train_data.drop(['is_duplicate'], axis=1, inplace=True)

    # missing data
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # train_data = train_data.fillna(method='ffill')
    train_data = train_data.fillna(train_data.median())

    # rebalance data
    pos_train = train_data[y_train == 1]
    neg_train = train_data[y_train == 0]
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    x_train = pd.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    params = {}
    params['nthread'] = 4
    params['seed'] = seed
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['silent'] = 1

    params['eta'] = 0.05
    params['max_depth'] = 6
    params['gamma'] = 0.2
    params['subsample'] = 0.7
    params['colsample_bytree'] = 0.6
    params['min_child_weight'] = 3

    num_rounds = 2000

    bst = xgb.train(params, d_train, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=10)

    # fix xgb bug: https://github.com/dmlc/xgboost/issues/1238
    bst.save_model(model_file)
    bst = xgb.Booster(params)
    bst.load_model(model_file)

    test_data = pd.read_csv(test_data_file)
    sub = pd.DataFrame()
    sub['test_id'] = test_data['test_id'].values
    test_data.drop(['test_id', 'question1', 'question2'], axis=1, inplace=True)

    d_test = xgb.DMatrix(test_data)
    p_test = bst.predict(d_test)

    sub['is_duplicate'] = p_test
    sub.to_csv(submission_file, index=False)
