import argparse
import pandas as pd
import emoji
import sys
from sys import stderr
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from lxml import etree
from xml.dom import minidom
from lxml import etree
from features import *
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import re
from emoji import UNICODE_EMOJI
import pickle
import os
from features_fnsp import *


def iter_docs(author):
    author_attr = author.attrib
    doc_dict = author_attr.copy()
    doc_dict['text'] = [' '.join([doc.text for doc in author.iter('document')])]

    return doc_dict


def create_data_frame(input_folder):
    os.chdir(input_folder)
    all_xml_files = glob.glob("*.xml")

    temp_list_of_DataFrames = []
    text_Data = pd.DataFrame()
    for file in all_xml_files:
        etree = ET.parse(file) 
        doc_df = pd.DataFrame(iter_docs(etree.getroot()))
        doc_df['author_id'] = file[:-4]
        temp_list_of_DataFrames.append(doc_df)
    text_Data = pd.concat(temp_list_of_DataFrames, axis=0)

    return text_Data



def getArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input Directory Path", required=True)
    parser.add_argument("-o", "--output", help="Ouput Directory Path", required=True)
    args = parser.parse_args()

    print("input {} output {} ".format(
        args.input,
        args.output,
    ))

    return args.input, args.output


def writefiles(data, output, lang):
    try:
        os.mkdir(output)
    except Exception as e:
        print(e)
    try:
        os.chdir(output)
    except Exception as e:
        print(e)
    try:
        os.mkdir(lang)
    except Exception as e:
        print(e)
    try:
        os.chdir(lang)
    except Exception as e:
        print(e)

    for index, row in data.iterrows():
        print(row['author_id'], row['lang'], row['author'])
        root = ET.Element("author", id=row['author_id'], lang=row['lang'], type=str(row['author']))
        tree = ET.ElementTree(root)
        tree.write(row['author_id'] + ".xml")

def runWithLang(input_folder, output_folder,lang):
    input_folder = os.path.join(input_folder,lang)
    data = create_data_frame(input_folder)

    preprocess(data)
    #wordvectorize(data)
    #charvectorize(data)

    if data.isnull().values.any():
        data.isnull().values.any()
        data.fillna(0, inplace=True)
    try:
        os.chdir(root)
    except Exception as e:
        print(e)
    print(os.getcwd())
    authormodel = pickle.load(open(os.path.join('models',lang,'modelSpreaders'), 'rb'))

    author = authormodel.predict(data.drop(['lang', 'text', 'author_id'], axis=1))
    data['author'] = author
    writefiles(data,output_folder,lang)


def main():
    global root
    root = os.getcwd()
    input_folder,output_folder = getArg()

    output_folder = os.path.join(output_folder)

    #runWithLang(input_folder, output_folder,'en')
    runWithLang(input_folder, output_folder,'es')


if __name__ == "__main__":
    main()



