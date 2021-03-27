import argparse
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import re
from scipy.stats import uniform as sp_rand
import numpy as np
import emoji
from emoji import UNICODE_EMOJI
from nltk.tokenize import TweetTokenizer
import pickle
from features_fnsp import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

def iter_docs(author):
    author_attr = author.attrib
    doc_dict = author_attr.copy()
    #    print(doc_dict)
    doc_dict['text'] = [' '.join([doc.text for doc in author.iter('document')])]

    return doc_dict


def create_data_frame(input_folder):
    os.chdir(input_folder)
    all_xml_files = glob.glob("*.xml")
    truth_data = pd.read_csv('truth.txt', sep=':::', names=['author_id', 'spreader'])

    temp_list_of_DataFrames = []
    text_Data = pd.DataFrame()
    for file in all_xml_files:
        etree = ET.parse(file)
        doc_df = pd.DataFrame(iter_docs(etree.getroot()))
        doc_df['author_id'] = file[:-4]
        temp_list_of_DataFrames.append(doc_df)
    text_Data = pd.concat(temp_list_of_DataFrames, axis=0)

    data = text_Data.merge(truth_data, on='author_id')
    return data


def getArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input directory path", required=True)
    parser.add_argument("-o", "--output", help="ouput directory path" )
    args = parser.parse_args()

    print("input {} output {} ".format(
        args.input,
        args.output,
    ))

    return args.input, args.output


def Model(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    LogisticRegression = LogisticRegression()
    from sklearn.ensemble import RandomForestClassifier
    RandomForestClassifier = RandomForestClassifier()
    from sklearn.naive_bayes import MultinomialNB
    MultinomialNB = MultinomialNB()
    from sklearn.svm import SVC 
    SVC = SVC(kernel='linear', gamma='scale')
    from sklearn.ensemble import GradientBoostingClassifier
    gb_clf = GradientBoostingClassifier()

    models = {'LogisticRegression': LogisticRegression,
               'RandomForestClassifier': RandomForestClassifier,
               'MultinomialNB': MultinomialNB,
               'SVC': SVC,
               'GBC': gb_clf}
    
    predictions = {}
    accuracy = {}

    for model in models:
        print(models[model])
        models[model].fit(X_train, y_train)
        predictions[model] = models[model].predict(X_test)
        accuracy[model] = accuracy_score(y_test, predictions[model])

    print('Best Model', max(accuracy, key=accuracy.get))
    print(classification_report(y_test, predictions[max(accuracy, key=accuracy.get)]))
    model = models[max(accuracy, key=accuracy.get)]
    return model


def buildModels(model, features, classLabel, modelname,lang):
    model.fit(features, classLabel)
    print(root)
    try:
        os.chdir(root)
        print('Change current Dir to ' + root)
    except Exception as e:
        print(e)

    try:

        os.mkdir('models')
        print('Make Dir to models')
    except Exception as e:
        print(e)

    try:
        os.chdir('models')
        print('Change current Dir to models')
    except Exception as e:
        print(e)

    try:
        os.mkdir(lang)
        print('Make Dir '+lang)

    except Exception as e:
        print(e)
    try:
        os.chdir(lang)
        print('Change current Dir to '+lang)

    except Exception as e:
        print(e)

    print('Saving model')
    pickle.dump(model, open(modelname, 'wb'))

    try:
        os.chdir(root)
        print('Change current Dir to '+root)

    except Exception as e:
        print(e)


def Language(input_folder,lang):
    input_folder = os.path.join(input_folder,lang)
    data = create_data_frame(input_folder)

    preprocess(data)
    #wordvectorize(data)
    #charvectorize(data)    
    
    if data.isnull().values.any():
        data.isnull().values.any()
        data.fillna(0, inplace=True)


    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['lang', 'text', 'author_id', 'spreader'], axis=1), data['spreader'], test_size=0.3, random_state=128, shuffle=True)

    model = Model(X_train, X_test, y_train, y_test)

    features = data.drop(['lang', 'text', 'author_id', 'spreader'], axis=1)

    classLabel = data['spreader']
    
    print('Building model for To Be a spreader or NOT to Be')
    
    buildModels(model, features, classLabel, 'modelSpreaders',lang)


def main():
    global root

    root = os.getcwd()

    input_folder, output_folder = getArg()

    Language(input_folder,'en')
    Language(input_folder,'es')


if __name__ == "__main__":
    main()
