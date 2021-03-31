# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 01:02:18 2020

@author: Bosko
"""

import argparse
import tfidf_tira
import extended_tfidf
import fullbatch_tfidf
import tira_sep_lang
import config
import os


def evaluate(path, path_out):
    """
    path_en = os.path.join(path, 'en')
    path_out_en = os.path.join(path_out, 'extended_tfidf/en')
    os.mkdir(path_out_en)
    extended_tfidf.fit(path_en, path_out_en, 'en')
    path_es = os.path.join(path, 'es')
    path_out_es = os.path.join(path_out, 'extended_tfidf/es')
    os.mkdir(path_out_es)
    extended_tfidf.fit(path_es, path_out_es, 'es')

    path_en = os.path.join(path, 'en')
    path_out_en = os.path.join(path_out, 'fullbatch_tfidf/en')
    os.mkdir(path_out_en)
    fullbatch_tfidf.fit(path_en, path_out_en, 'en')
    path_es = os.path.join(path, 'es')
    path_out_es = os.path.join(path_out, 'fullbatch_tfidf/es')
    os.mkdir(path_out_es)
    fullbatch_tfidf.fit(path_es, path_out_es, 'es')
    """

    path_en = os.path.join(path, 'en')
    path_out_en = os.path.join(path_out, 'tira_sep_lang/en')
    os.mkdir(path_out_en)
    tira_sep_lang.fit(path_en, path_out_en, 'en')
    path_es = os.path.join(path, 'es')
    path_out_es = os.path.join(path_out, 'tira_sep_lang/es')
    os.mkdir(path_out_es)
    tira_sep_lang.fit(path_es, path_out_es, 'es')

    """
    path_en = os.path.join(path, 'en')
    path_out_en = os.path.join(path_out, 'tfidf_tira/en')
    os.mkdir(path_out_en)
    tfidf_tira.fit(path_en, path_out_en, 'en')
    path_es = os.path.join(path, 'es')
    path_out_es = os.path.join(path_out, 'tfidf_tira/es')
    os.mkdir(path_out_es)
    tfidf_tira.fit(path_es, path_out_es, 'es')
    """


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='Predict task.')
    parser.add_argument('-i', dest='input_dir', metavar='i', type=str, nargs=1,
                        help='Input directory.', required=True)
    parser.add_argument('-o', dest='output_dir', metavar='o', type=str, nargs=1,
                        help='Output directory.', required=True)
    args = parser.parse_args()
    print(args.input_dir)
    """
    input_dir = "../../../datasets/pan20-fake-test"
    output_dir = "../../../output/koloski-fake"
    evaluate(input_dir, output_dir)
