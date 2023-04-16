import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json
# from bson import json_util
import csv


def loadData(pos_path, neg_path):
    # Load positive tweets
    sentences_pos = []
    with open(pos_path) as pos:
        for line in pos:
            sentences_pos.append(line)
    df_pos = pd.DataFrame(sentences_pos, columns=['tweet'])

    # Load negative tweets
    sentences_neg = []
    with open(neg_path) as neg:
        for line in neg:
            sentences_neg.append(line)
    df_neg = pd.DataFrame(sentences_neg, columns=['tweet'])

    # include sentiment labels
    df_pos.drop_duplicates(subset='tweet', keep='first', inplace=True)
    df_pos['sentiment'] = 1
    df_neg.drop_duplicates(subset='tweet', keep='first', inplace=True)
    df_neg['sentiment'] = 0

    # concatenate the positive and negative DataFrames
    df = pd.concat([df_pos, df_neg])
    df.reset_index(inplace=True, drop=True)
    return df


def save_json_result(model_name, result):
    """
    Save json to a directory and a filename. Used for saving results of the cross-validation 
    :param model_name:
    :param result:
    :return:
    """
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    with open(os.path.join("results/", result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

