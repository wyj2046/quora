# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from scipy.stats import randint, uniform
from sklearn.externals import joblib


seed = 2017
np.random.seed(seed)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "usage:%s train_data model" % sys.argv[0]
        sys.exit(-1)
    train_data_file = sys.argv[1]
    model_file = sys.argv[2]

    train_data = pd.read_csv(train_data_file)
    train_data.drop(['id', 'qid1', 'qid2', 'question1', 'question2'], axis=1, inplace=True)
    y_train = train_data['is_duplicate'].values
    train_data.drop(['is_duplicate'], axis=1, inplace=True)
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data = train_data.fillna(train_data.median())
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

    params_fixed = {}
    params_fixed['nthread'] = 2
    params_fixed['seed'] = seed
    params_fixed['silent'] = True
    params_fixed['objective'] = 'binary:logistic'

    params_dist_grid = {}
    params_dist_grid['max_depth'] = range(2, 7, 1)
    params_dist_grid['gamma'] = [i / 10.0 for i in range(0, 5)]
    params_dist_grid['n_estimators'] = [500, 1000, 1500, 2000]
    params_dist_grid['learning_rate'] = [0.02, 0.03, 0.04, 0.05]
    params_dist_grid['subsample'] = [i / 10.0 for i in range(6, 10)]
    params_dist_grid['colsample_bytree'] = [i / 10.0 for i in range(6, 10)]
    params_dist_grid['min_child_weight'] = range(1, 10, 2)

    # rs_grid = RandomizedSearchCV(
    #     estimator=xgb.XGBClassifier(**params_fixed),
    #     param_distributions=params_dist_grid,
    #     n_iter=10,
    #     cv=3,
    #     scoring='neg_log_loss',
    #     random_state=seed,
    #     n_jobs=8,
    #     verbose=2
    # )

    rs_grid = GridSearchCV(
        xgb.XGBClassifier(**params_fixed),
        params_dist_grid,
        cv=3,
        scoring='neg_log_loss',
        n_jobs=8,
        verbose=2
    )

    rs_grid.fit(x_train, y_train)

    print rs_grid.grid_scores_
    print 'BEST', rs_grid.best_params_, rs_grid.best_score_

    joblib.dump(rs_grid.best_estimator_, model_file)
