import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaModel
import matplotlib.pyplot as plt
import pickle
from keras.models import model_from_json

# Test data
TEST_PATH = "data/test_data.txt"

# Load the test set.
test_sent = []
idx = []
with open(TEST_PATH) as test:
    for line in test:
        split = line.split(",", 1)
        idx.append(int(split[0]))
        test_sent.append(split[1])

data = {'index':idx,'tweet':test_sent}
df_test = pd.DataFrame(data)
df_test.head()

# RoBERTa
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base')


def batch_encode(tokenizer, texts, batch_size=256, max_length=55):
    """""""""
    This function encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed
    into a pre-trained transformer model.

    Input:
        - tokenizer:   okenizer: tokenizer to be used to tokenize the sentences (related to the model)
        - texts:       List containin
        - batch_size:  Number of sentences to process simultaneously (default-256)
        - max_length:  Maximum number of tokens per sentence (default=55)
    Output:
        - input_ids:       tensor representing the set of indeces for each sentence
        - attention_mask:  tensor representing the attention masks fro each sentence
    """""""""

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])


    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)

# encode the test sentences the same way we did for the training data
test_ids, test_attention = batch_encode(tokenizer_roberta, df_test.tweet.to_list())

# Load the saved model and its weights
json_file = open('RoBERTa_finetuned_MLP.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={'TFRobertaModel': TFRobertaModel})
loaded_model.load_weights('./RoBERTa_finetuned_MLP_weights.h5')

# Create prediction vector
pred = loaded_model.predict([test_ids, test_attention])
pred[pred>0.5] = 1
pred[pred<=0.5] = -1

# define the function to create submission
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

# create submission
OUTPUT_PATH = 'submission_to_AIcrowd.csv'
create_csv_submission(df_test['index'], pred, OUTPUT_PATH)
