import pandas as pd
import os
import glob
import argparse
import shutil
import random

"""
Creates test and training sets and matching truth.txt

Args: -i path to data sets directory, -s size of data set
"""

if __name__ == "__main__":

    owd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input Directory Path", required=True)
    parser.add_argument("-s", "--size", help="Size of data set", required=True)
    args = parser.parse_args()

    split_size = int(int(args.size) * 0.8)

    en_input = args.input + '/en'
    os.chdir(en_input)

    truth_data_en = pd.read_csv('truth.txt', sep=':::', names=['author_id', 'spreader'])
    en_full_truth = open("truth.txt")

    os.mkdir("entrain")
    os.mkdir("entest")

    en_truth_train = open("entrain/truth.txt", "x")
    en_truth_test = open("entest/truth.txt", "x")

    all_xml_files = glob.glob("*.xml")
    random.shuffle(all_xml_files)
    test_xml = all_xml_files[split_size:]
    train_xml = all_xml_files[:split_size]

    for file in train_xml:
        target = "entrain/" + file
        shutil.copyfile(file, target)
        id = file[:-4]
        type = str(truth_data_en.loc[truth_data_en['author_id'] == id, 'spreader'].iloc[0])
        en_truth_train.write(id+':::'+type+ '\n')
    for file in test_xml:
        target = "entest/" + file
        shutil.copyfile(file, target)
        id = file[:-4]
        type = str(truth_data_en.loc[truth_data_en['author_id'] == id, 'spreader'].iloc[0])
        en_truth_test.write(id+':::'+type+ '\n')

    en_truth_test.close()
    en_truth_train.close()
    en_full_truth.close()

    # ES Begin
    os.chdir(owd)

    es_input = args.input + '/es'

    os.chdir(es_input)
    os.mkdir("estrain")
    os.mkdir("estest")

    truth_data_es = pd.read_csv('truth.txt', sep=':::', names=['author_id', 'spreader'])
    es_full_truth = open("truth.txt")
    es_truth_train = open("estrain/truth.txt", "x")
    es_truth_test = open("estest/truth.txt", "x")

    all_xml_files2 = glob.glob("*.xml")
    random.shuffle(all_xml_files2)
    test_xml2 = all_xml_files2[split_size:]
    train_xml2 = all_xml_files2[:split_size]

    for file in train_xml2:
        target = "estrain/" + file
        shutil.copyfile(file, target)
        id = file[:-4]
        type = str(truth_data_es.loc[truth_data_es['author_id'] == id, 'spreader'].iloc[0])
        es_truth_train.write(id + ':::' + type+ '\n')
    for file in test_xml2:
        target = "estest/" + file
        shutil.copyfile(file, target)
        id = file[:-4]
        type = str(truth_data_es.loc[truth_data_es['author_id'] == id, 'spreader'].iloc[0])
        es_truth_test.write(id + ':::' + type + '\n')
    es_truth_test.close()
    es_truth_train.close()
    es_full_truth.close()






