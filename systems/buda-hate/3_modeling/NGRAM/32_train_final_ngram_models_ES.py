import pickle
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

data_v1_es = pd.read_csv('../../data/clean_es_data_v1.tsv', delimiter='\t', encoding='utf-8')
data_v2_es = pd.read_csv('../../data/clean_es_data_v2.tsv', delimiter='\t', encoding='utf-8')

# LR
lr_vectorizer_v1 = TfidfVectorizer(min_df=7, ngram_range=(2, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)
lr_X_v1 = lr_vectorizer_v1.fit_transform(data_v2_es["Tweets"])
pickle.dump(lr_vectorizer_v1, open('../../models/es/lr_vectorizer_v1_es.pickle', 'wb'))
lr_v1 = LogisticRegression(C=1000, penalty='l2', solver='liblinear', fit_intercept=False, verbose=0)
lr_v1.fit(lr_X_v1, data_v1_es['spreader'])
pickle.dump(lr_v1, open('../../models/es/lr_v1_es.sav', 'wb'))

# SVM
svm_vectorizer_v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=6, sublinear_tf=True, use_idf=True, smooth_idf=True)
svm_X_v1 = svm_vectorizer_v1.fit_transform(data_v2_es["Tweets"])
pickle.dump(svm_vectorizer_v1, open('../../models/es/svm_vectorizer_v1_es.pickle', 'wb'))
svm_v1 = svm.SVC(C=10, kernel='linear', verbose=False, probability=True)
svm_v1.fit(svm_X_v1, data_v1_es['spreader'])
pickle.dump(svm_v1, open('../../models/es/svm_v1_es.sav', 'wb'))

# RF
rf_vectorizer_v1 = TfidfVectorizer(min_df=8, ngram_range=(2, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)
rf_X_v1 = rf_vectorizer_v1.fit_transform(data_v1_es['Tweets'])
pickle.dump(rf_vectorizer_v1, open('../../models/es/rf_vectorizer_v1_es.pickle', 'wb'))
rf_v1 = RandomForestClassifier(n_estimators=400, min_samples_leaf=7, criterion='gini')
rf_v1.fit(rf_X_v1, data_v1_es["spreader"])
pickle.dump(rf_v1, open('../../models/es/rf_v1_es.sav', 'wb'))

# XGB
vect_xgb_es_v1 = TfidfVectorizer(min_df=7, ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)
xgb_X_v1 = vect_xgb_es_v1.fit_transform(data_v1_es['Tweets'])
pickle.dump(vect_xgb_es_v1, open('../../models/es/vect_xgb_es_v1.pickle', 'wb'))
xgb_es_v1 = xgb.XGBClassifier(colsample_bytree=0.5, eta=0.1, max_depth=4, n_estimators=300, subsample=0.7)
xgb_es_v1.fit(xgb_X_v1, data_v1_es['spreader'])
pickle.dump(xgb_es_v1, open('../../models/es/xgb_es_v1.sav', 'wb'))

