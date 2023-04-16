import pandas as pd
import numpy as np
import tensorflow as tf

            
def get_sent_embe(df, tokenizer, model):
    """""""""
    Obtains the sentence embedding.
    
    Input: 
        - df:         DataFrame containing the sentences to be embedded
        - tokenizer:  tokenizer to be used to tokenize the sentences (related to the model)
        - model:      TensorFlow BERT model to be used to obtain the embeddings
    Output: 
        - features_np:  numpy array of sentence embeddings 
    """""""""
    
    features = []
    
    # change if GPU not available
    with tf.device("/gpu:0"):

        tokenized = df['tweet'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded_tr = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
        attention_mask_tr = np.where(padded_tr != 0, 1, 0)
        print(attention_mask_tr.shape)

        input_ids_tr = tf.data.Dataset.from_tensor_slices(padded_tr).batch(200)
        attention_mask_tr = tf.data.Dataset.from_tensor_slices(attention_mask_tr).batch(200)

        iterator = iter(attention_mask_tr)
        count = 0
        for element in input_ids_tr:
            print("iteration: ", count)
            count += 1
            last_hidden_layer = model(element, attention_mask=iterator.get_next())[0][:, 0, :].numpy()
            features.append(last_hidden_layer)

        features_np = np.vstack(features)

    return features_np


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


def batch_encode_cross_validation(tokenizer, texts, batch_size=128, max_length=55):
    """""""""
    This function encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks.

    Input:
        - tokenizer:   okenizer: tokenizer to be used to tokenize the sentences (related to the model)
        - texts:       List containin
        - batch_size:  Number of sentences to process simultaneously (default-256)
        - max_length:  Maximum number of tokens per sentence (default=55)
    Output:
        - input_ids:       Pandas DataFrame representing the set of indeces for each sentence
        - attention_mask:  Pandas DataFrame representing the attention masks fro each sentence
    """""""""
    
    
    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             #padding='longest', #implements dynamic padding
                                             padding='max_length',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])
    
    # return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)
    return pd.DataFrame(input_ids), pd.DataFrame(attention_mask)