#############################################
#                                           #
#              importing packages           #
#                                           #
#############################################
from emoji import UNICODE_EMOJI
from html import unescape
import joblib
from lexical_diversity import lex_div as ld
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import random
import re
from statistics import pstdev

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import sys
import xgboost as xgb
import xml.etree.ElementTree as ET

###############################################
#                                             #
#        defining auxiliary functions         #
#                                             #
###############################################


# text cleaning v1
def cleaning_v1(tweet_lista):
    cleaned_feed_v1 = []
    for feed in tweet_lista:
        feed = feed.lower()
        feed = re.sub('[^0-9a-z #@]', "", feed)
        feed = re.sub('[\n]', " ", feed)
        cleaned_feed_v1.append(feed)
    return cleaned_feed_v1


# emoji handling
def is_emoji(s):
    return s in UNICODE_EMOJI


def emoji_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()


# text cleaning v2
def cleaning_v2(tweet_lista):
    cleaned_feed_v2 = []
    for feed in tweet_lista:
        feed = feed.lower()
        feed = emoji_space(feed)
        feed = re.sub('[,.\'\"‘’”“]', '', feed)
        feed = re.sub(r'([a-z\'-’]+)', r'\1 ', feed)
        feed = re.sub(r'(?<![?!:;/])([:\'\";.,?()/!])(?= )', '', feed)
        feed = re.sub('[\n]', ' ', feed)
        feed = ' '.join(feed.split())
        cleaned_feed_v2.append(feed)
    return cleaned_feed_v2


# setting save and load location parameters
if __name__ == "__main__":
    a = sys.argv[2]
    b = sys.argv[4]
    a = "../../../datasets/pan20-fake-test"
    b = "../../../output/buda-fake"
    print(a)
    print(b)
    # location of used model files
    model_loc = "../models/"

    # dictionary for prediction
    spreader = {'non-spreader': 0, 'spreader': 1}

    # setting up target directories
    os.mkdir(b+'/en/')
    os.mkdir(b+'/es/')

    ###############################################
    ###############################################
    #                                             #
    #                  ENGLISH                    #
    #                                             #
    ###############################################
    ###############################################

    # read English data
    pathlist = Path(a+"/en/").glob('*.xml')
    ids = []  # id list for concatenated list of tweets of each user
    ids_tw = []  # id list for tweetwise list
    x_test = []  # string list, 1 concatenated item per user
    x_raw = []  # string list, 1 item per tweet
    for path in pathlist:  # iterate files
        head, tail = os.path.split(path)
        t = tail.split(".")
        author = t[0]
        ids.append(author)
        path_in_str = str(path)
        tree = ET.parse(path_in_str)
        root = tree.getroot()
        for child in root:
            xi = []
            for ch in child:
                xi.append(ch.text)
                x_raw.append(unescape(ch.text))  # recode xml char-s
                ids_tw.append(author)
            content = ' '.join(xi)
            x_test.append(content)

    # dataframe for text cleaning v1 for n-gram models
    en_data_v1 = pd.DataFrame()
    en_data_v1["ID"] = ids
    en_data_v1["Tweets"] = x_test

    # DF for text cleaning v2 for n-gram models
    en_data_v2 = pd.DataFrame()
    en_data_v2["ID"] = ids
    en_data_v2["Tweets"] = x_test

    # tweetwise df without further cleaning for descriptive stat model
    en_data_tw = pd.DataFrame()
    en_data_tw["ID"] = ids_tw
    en_data_tw["Tweets"] = x_raw

    #############################################
    #                                           #
    #              N-gram models                #
    #                                           #
    #############################################
    # clean English data
    feed_list = en_data_v1["Tweets"].tolist()
    en_data_v1["Tweets"] = cleaning_v1(feed_list)
    en_data_v2["Tweets"] = cleaning_v2(feed_list)

    # load vectorizers
    vect_rf = pickle.load(open(model_loc + 'en\\rf_vectorizer_v1.pickle', "rb"))
    X_test_en_RF = vect_rf.transform(en_data_v1["Tweets"])

    vect_SVM = pickle.load(open(model_loc + 'en\\svm_vectorizer_v1.pickle', "rb"))
    X_test_en_SVM = vect_SVM.transform(en_data_v1["Tweets"])

    vect_LR = pickle.load(open(model_loc + 'en\\lr_vectorizer_v1.pickle', "rb"))
    X_test_en_LR = vect_LR.transform(en_data_v1["Tweets"])

    vect_XGB = pickle.load(open(model_loc + 'en\\vect_xgb_en_v1.pickle', "rb"))
    X_test_en_XGB = vect_XGB.transform(en_data_v2["Tweets"])

    # load models
    # RF
    en_RF = pickle.load(open(model_loc + 'en\\rf_v2.sav', "rb"))
    # SVM
    en_SVM = pickle.load(open(model_loc + 'en\\svm_v1.sav', "rb"))
    # LR
    en_LR = pickle.load(open(model_loc + 'en\\lr_v1.sav', "rb"))
    # XGB
    en_XGB = pickle.load(open(model_loc + 'en\\xgb_en_v1.sav', "rb"))

    # predicting
    en_preds_RF = en_RF.predict_proba(X_test_en_RF)[:, 1]
    en_preds_SVM = en_SVM.predict_proba(X_test_en_SVM)[:, 1]
    en_preds_LR = en_LR.predict_proba(X_test_en_LR)[:, 1]
    en_preds_XGB = en_XGB.predict_proba(X_test_en_XGB)[:, 1]

    #############################################
    #                                           #
    #         descriptive stat models           #
    #                                           #
    #############################################
    #
    # feature extraction
    #

    # initialize feature DF with ID-s per author
    en_data_tweet_consist = pd.DataFrame(list(zip([en_data_tw["ID"][i*100] for
                                                   i in range(int(len(en_data_tw["ID"])/100))])), columns=['ID'])

    # length stats of tweets per author (character & word)
    len_tw_char = [len(i) for i in en_data_tw["Tweets"]]
    len_tw_word = [len(i.split(" ")) for i in en_data_tw["Tweets"]]

    # SD
    len_char_sd_auth = [pstdev(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
    len_word_sd_auth = [pstdev(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

    # min - max - range - mean
    len_char_min_auth = [min(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
    len_word_min_auth = [min(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

    len_char_max_auth = [max(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
    len_word_max_auth = [max(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

    len_char_rng_auth = [max(len_tw_char[i*100:i*100+99])-min(len_tw_char[i*100:i*100+99]) for
                         i in range(int(len(len_tw_char)/100))]
    len_word_rng_auth = [max(len_tw_word[i*100:i*100+99])-min(len_tw_word[i*100:i*100+99]) for
                         i in range(int(len(len_tw_word)/100))]

    len_char_mean_auth = [np.mean(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
    len_word_mean_auth = [np.mean(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

    # vocab variety
    tweets_szerz = [" ".join(list(en_data_tw["Tweets"])[i*100:99+i*100]) for
                    i in range(int(len(len_tw_char)/100))]  # concatenate tweets of each author to calculate TTR

    ttr_szerz = [ld.ttr(ld.flemmatize(i)) for i in tweets_szerz]

    # number of tags
    # RT
    rt_szerz = [np.sum([k == "RT" for k in i.split(" ")]) for i in tweets_szerz]
    # URL
    url_szerz = [np.sum([k == "#URL#" for k in i.split(" ")]) for i in tweets_szerz]
    # hashtag
    hsg_szerz = [np.sum([k == "#HASHTAG#" for k in i.split(" ")]) for i in tweets_szerz]
    # user
    user_szerz = [np.sum([k == "#USER#" for k in i.split(" ")]) for i in tweets_szerz]
    # ...
    p_szerz = [np.sum([k[-1:] == "…" for k in i.split(" ")]) for i in tweets_szerz]
    # emoji
    emoj_szerz = []
    for aut in tweets_szerz:
        emdb = 0
        for tok in aut.split(" "):
            for c in tok:
                emdb += c in UNICODE_EMOJI
        emoj_szerz.append(emdb)

    # populate feature DF
    en_data_tweet_consist["len_char_sd_auth"] = len_char_sd_auth
    en_data_tweet_consist["len_word_sd_auth"] = len_word_sd_auth

    en_data_tweet_consist["len_char_min_auth"] = len_char_min_auth
    en_data_tweet_consist["len_word_min_auth"] = len_word_min_auth

    en_data_tweet_consist["len_char_max_auth"] = len_char_max_auth
    en_data_tweet_consist["len_word_max_auth"] = len_word_max_auth

    en_data_tweet_consist["len_char_rng_auth"] = len_char_rng_auth
    en_data_tweet_consist["len_word_rng_auth"] = len_word_rng_auth

    en_data_tweet_consist["len_char_mean_auth"] = len_char_mean_auth
    en_data_tweet_consist["len_word_mean_auth"] = len_word_mean_auth

    en_data_tweet_consist["rt_szerz"] = rt_szerz
    en_data_tweet_consist["url_szerz"] = url_szerz
    en_data_tweet_consist["hsg_szerz"] = hsg_szerz
    en_data_tweet_consist["user_szerz"] = user_szerz
    en_data_tweet_consist["p_szerz"] = p_szerz
    en_data_tweet_consist["emoj_szerz"] = emoj_szerz
    en_data_tweet_consist["ttr_szerz"] = ttr_szerz

    #############################################
    #
    # descriptive stat model prediction
    #
    xgb_tweetcons_en = joblib.load(model_loc + "en\\tweetconsistence_xgboost_en_v2")
    en_twcons_pred = xgb_tweetcons_en.predict_proba(en_data_tweet_consist.iloc[:, 1:])[:, 1]
    print(en_data_tweet_consist)

    #############################################
    #                                           #
    #              stacking models              #
    #                                           #
    #############################################
    results_en = pd.DataFrame()

    results_en["lr"] = en_preds_LR
    results_en["svm"] = en_preds_SVM
    results_en["rf"] = en_preds_RF
    results_en["xgb"] = en_preds_XGB
    results_en["xgb_tw"] = en_twcons_pred

    # load model
    en_ensemble = joblib.load(model_loc + "en\\ensemble_en_linreg")
    # predictions
    en_preds = en_ensemble.predict(results_en)

    # Saving predictions for ENGLISH DATA
    for i in range(len(ids)):
        a_id = str(ids[i])
        pred = en_preds[i]   
        root = ET.Element("author", id=a_id, lang="en", type=str(pred))
        tree = ET.ElementTree(root)
        tree.write(b + "/en/" + a_id + ".xml")

    ###############################################
    ###############################################
    #                                             #
    #                  SPANISH                    #
    #                                             #
    ###############################################
    ###############################################

    # read in Spanish data
    pathlist = Path(a+"/es/").glob('*.xml')
    ids_es = []  # id list for concatenated list of tweets of each user
    ids_tw = []  # id list for tweetwise list
    x_test_es = []  # string list, 1 concatenated item per user
    x_raw = []  # string list, 1 item per tweet
    for path in pathlist:  # iterate files
        head, tail = os.path.split(path)
        t = tail.split(".")
        author = t[0]
        ids_es.append(author)
        path_in_str = str(path)
        tree = ET.parse(path_in_str)
        root = tree.getroot()
        for child in root:
            xi = []
            for ch in child:
                xi.append(ch.text)
                x_raw.append(unescape(ch.text))  # recode xml char-s
                ids_tw.append(author)
            content = ' '.join(xi)
            x_test_es.append(content)

    # DF for n-gram models
    es_data = pd.DataFrame()
    es_data["ID"] = ids_es
    es_data["Tweets"] = x_test_es

    # DF for text cleaning v2 for n-gram models
    es_data_v2 = pd.DataFrame()
    es_data_v2["ID"] = ids
    es_data_v2["Tweets"] = x_test

    # tweetwise df without further cleaning for descriptive stat model
    es_data_tw = pd.DataFrame()
    es_data_tw["ID"] = ids_tw
    es_data_tw["Tweets"] = x_raw

    #############################################
    #                                           #
    #               N-gram models               #
    #                                           #
    #############################################
    # clean Spanish data
    feed_list = es_data["Tweets"].tolist()
    es_data["Tweets"] = cleaning_v1(feed_list)
    es_data_v2["Tweets"] = cleaning_v2(feed_list)

    # load vectorizers
    vect_rf = pickle.load(open(model_loc + 'es\\rf_vectorizer_v1_es.pickle', "rb"))
    X_test_es_RF = vect_rf.transform(es_data["Tweets"])

    vect_SVM = pickle.load(open(model_loc + 'es\\svm_vectorizer_v1_es.pickle', "rb"))
    X_test_es_SVM = vect_SVM.transform(es_data_v2["Tweets"])

    vect_LR = pickle.load(open(model_loc + 'es\\lr_vectorizer_v1_es.pickle', "rb"))
    X_test_es_LR = vect_LR.transform(es_data["Tweets"])

    vect_XGB = pickle.load(open(model_loc + 'es\\vect_xgb_es_v1.pickle', "rb"))
    X_test_es_XGB = vect_XGB.transform(es_data["Tweets"])

    # load models
    # RF
    es_RF = pickle.load(open(model_loc + 'es\\rf_v1_es.sav', "rb"))
    # SVM
    es_SVM = pickle.load(open(model_loc + 'es\\svm_v1_es.sav', "rb"))
    # LR
    es_LR = pickle.load(open(model_loc + 'es\\lr_v1_es.sav', "rb"))
    # XGB
    es_XGB = pickle.load(open(model_loc + 'es\\xgb_es_v1.sav', "rb"))

    # predicting
    es_preds_RF = es_RF.predict_proba(X_test_es_RF)[:, 1]
    es_preds_SVM = es_SVM.predict_proba(X_test_es_SVM)[:, 1]
    es_preds_LR = es_LR.predict_proba(X_test_es_LR)[:, 1]
    es_preds_XGB = es_XGB.predict_proba(X_test_es_XGB)[:, 1]

    #############################################
    #                                           #
    #         descriptive stat models           #
    #                                           #
    #############################################
    #
    # feature extraction
    #

    # initialize feature DF with ID-s per author
    es_data_tweet_consist = pd.DataFrame(list(zip([es_data_tw["ID"][i*100] for
                                                   i in range(int(len(es_data_tw["ID"])/100))])), columns=['ID'])

    # length of tweets per author
    len_tw_char = [len(i) for i in es_data_tw["Tweets"]]
    len_tw_word = [len(i.split(" ")) for i in es_data_tw["Tweets"]]

    # SD
    len_char_sd_auth = [pstdev(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
    len_word_sd_auth = [pstdev(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

    # min - max - range - mean
    len_char_min_auth = [min(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
    len_word_min_auth = [min(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

    len_char_max_auth = [max(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
    len_word_max_auth = [max(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

    len_char_rng_auth = [max(len_tw_char[i*100:i*100+99])-min(len_tw_char[i*100:i*100+99]) for
                         i in range(int(len(len_tw_char)/100))]
    len_word_rng_auth = [max(len_tw_word[i*100:i*100+99])-min(len_tw_word[i*100:i*100+99]) for
                         i in range(int(len(len_tw_word)/100))]

    len_char_mean_auth = [np.mean(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
    len_word_mean_auth = [np.mean(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

    # vocab variety
    tweets_szerz = [" ".join(list(es_data_tw["Tweets"])[i*100:99+i*100]) for
                    i in range(int(len(len_tw_char)/100))]
    ttr_szerz = [ld.ttr(ld.flemmatize(i)) for i in tweets_szerz]

    # number of tags
    # RT
    rt_szerz = [np.sum([k == "RT" for k in i.split(" ")]) for i in tweets_szerz]
    # URL
    url_szerz = [np.sum([k == "#URL#" for k in i.split(" ")]) for i in tweets_szerz]
    # hashtag
    hsg_szerz = [np.sum([k == "#HASHTAG#" for k in i.split(" ")]) for i in tweets_szerz]
    # user
    user_szerz = [np.sum([k == "#USER#" for k in i.split(" ")]) for i in tweets_szerz]
    # ...
    p_szerz = [np.sum([k[-1:] == "…" for k in i.split(" ")]) for i in tweets_szerz]
    # emojis
    emoj_szerz = []
    for aut in tweets_szerz:
        emdb = 0
        for tok in aut.split(" "):
            for c in tok:
                emdb += c in UNICODE_EMOJI
        emoj_szerz.append(emdb)

    # populate feature DF
    es_data_tweet_consist["len_char_sd_auth"] = len_char_sd_auth
    es_data_tweet_consist["len_word_sd_auth"] = len_word_sd_auth

    es_data_tweet_consist["len_char_min_auth"] = len_char_min_auth
    es_data_tweet_consist["len_word_min_auth"] = len_word_min_auth

    es_data_tweet_consist["len_char_max_auth"] = len_char_max_auth
    es_data_tweet_consist["len_word_max_auth"] = len_word_max_auth

    es_data_tweet_consist["len_char_rng_auth"] = len_char_rng_auth
    es_data_tweet_consist["len_word_rng_auth"] = len_word_rng_auth

    es_data_tweet_consist["len_char_mean_auth"] = len_char_mean_auth
    es_data_tweet_consist["len_word_mean_auth"] = len_word_mean_auth

    es_data_tweet_consist["rt_szerz"] = rt_szerz
    es_data_tweet_consist["url_szerz"] = url_szerz
    es_data_tweet_consist["hsg_szerz"] = hsg_szerz
    es_data_tweet_consist["user_szerz"] = user_szerz
    es_data_tweet_consist["p_szerz"] = p_szerz
    es_data_tweet_consist["emoj_szerz"] = emoj_szerz
    es_data_tweet_consist["ttr_szerz"] = ttr_szerz

    #############################################
    #
    # descriptive stat model prediction
    #
    xgb_tweetcons_es = joblib.load(model_loc + "es\\tweetconsistence_xgboost_es_v2")
    es_twcons_pred = xgb_tweetcons_es.predict_proba(es_data_tweet_consist.iloc[:, 1:])[:, 1]

    #############################################
    #                                           #
    #              stacking models              #
    #                                           #
    #############################################
    results_es = pd.DataFrame()

    results_es["lr"] = es_preds_LR
    results_es["svm"] = es_preds_SVM
    results_es["rf"] = es_preds_RF
    results_es["xgb"] = es_preds_XGB
    results_es["xgb_tw"] = es_twcons_pred

    # load model
    es_ensemble = pickle.load(open(model_loc + 'es\\es_aggregate_lr.sav', "rb"))
    # predictions
    es_preds = es_ensemble.predict(results_es)

    # Saving predictions for SPANISH DATA
    for i in range(len(ids_es)):
        a_id = str(ids_es[i])
        pred = es_preds[i]   
        root = ET.Element("author", id=a_id, lang="es", type=str(pred))  # abc sorrendbe teszi!!!!
        tree = ET.ElementTree(root)
        tree.write(b + "/es/" + a_id + ".xml")
