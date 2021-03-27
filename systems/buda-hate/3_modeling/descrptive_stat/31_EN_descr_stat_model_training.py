#########################################################
#                                                       #
#                   importing packages                  #
#                                                       #
#########################################################
import joblib
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# reading DF
en_data_tweet_consist = pd.read_pickle('../../data/en_data_tweet_consist.pkl')

########################################################
#                                                      #
#                   model training                     #
#                                                      #
#########################################################

#######
#
# benchmark with default setups on half training data
#

xgb_def = xgb.XGBClassifier()
xgb_def.fit(en_data_tweet_consist.iloc[80:, 2:],
            np.ravel(en_data_tweet_consist.iloc[80:, 1:2]))

# train data metrics
print(classification_report(en_data_tweet_consist.iloc[:80, 1:2],
                            xgb_def.predict(en_data_tweet_consist.iloc[:80, 2:])))
# test data metrics
print(classification_report(en_data_tweet_consist.iloc[80:, 1:2],
                            xgb_def.predict(en_data_tweet_consist.iloc[80:, 2:])))


########################
#
# hyperparameter search
#

# scoring metrics
f1_scorer = make_scorer(f1_score, average='weighted')
acc_scorer = make_scorer(accuracy_score)
scoring = {'F1': f1_scorer, 'Accuracy': acc_scorer}

# defining search space
params = {
        'min_child_weight': [2, 3, 4, 5],
        'gamma': [0, 1, 2, 4],
        'subsample': [0.6, 0.8, 1],
        'colsample_bytree': [0.8, 0.9, 1],
        'colsample_bynode': [0.8, 0.9, 1],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.3, 0.2, 0.1],
        'reg_alpha': [0.1, 0.3, 0.7],
        'n_estimators': [100, 150, 200]
        }

# grid search with CV
xgb_cv_clf = xgb.XGBClassifier(use_label_encoder=False)
xgb_cv = GridSearchCV(xgb_cv_clf,
                      param_grid=params,
                      scoring=scoring,
                      refit="Accuracy",
                      return_train_score=True,
                      cv=5,
                      verbose=100,
                      n_jobs=-1)
xgb_cv.fit(en_data_tweet_consist.iloc[:, 2:], np.ravel(en_data_tweet_consist.iloc[:, 1:2]).astype(int))

# performance metrics
print(xgb_cv.best_score_, xgb_cv.best_params_)

print(classification_report(np.ravel(en_data_tweet_consist.iloc[:, 1:2]).astype(int),
                            xgb_cv.best_estimator_.predict(en_data_tweet_consist.iloc[:, 2:])))

joblib.dump(xgb_cv.best_estimator_, "../../models/en/tweetconsistence_xgboost_en_v2")
