from nltk.tokenize.casual import TweetTokenizer
from tqdm import tqdm
import numpy as np

def feature(data,model1):
    
    tweet_matrix=[]
    for twe in tqdm(data):
        tweet_vec=[]
        for token in twe:
            if token in model1.wv:             #if token is not in lookuptable word vector is zero
                tweet_vec.append(model1.wv[token])
        tweet_vec=np.array([tweet_vec]).reshape(-1,model1.vector_size)
        #print(tweet_vec.shape)
        tweet_matrix.append(tweet_vec.mean(axis=0))
    tx=np.array([tweet_matrix]).reshape(-1,model1.vector_size)
    return tx

def tokenisation(tweets):
    data = []
    # iterate through each sentence in the file
    for i in tqdm(tweets):
        #print(i)
        #print("/n")
        tweet = []
        # tokenize the tweet into words
        for j in TweetTokenizer().tokenize(i):
            tweet.append(j.lower())
            #print(tweet)
  
        data.append(tweet)
    return data

def split_data(x, y, ratio, seed):
    # set seed
    np.random.seed(seed)
    # split the data based on the given ratio
    index=np.random.permutation(len(y))
    train_size=int(np.floor(ratio*len(y)))
    
    idx_train=index[: train_size]#trainings data indices
    idx_val=index[train_size :]#valdiation data indices
    
    y_train=y[idx_train]#trainings results
    y_val=y[idx_val]#validation results
    
    x_train=x[idx_train,:]#trainings data
    x_val=x[idx_val,:]#validation data
    return y_train, y_val, x_train, x_val

def take_part(tx,y,num_data,seed):
    #input is a np array
    np.random.seed(seed)
    indexs=np.random.permutation(len(tx))
    
    idx=indexs[: num_data]# smaller size
    tx_small=tx[idx,:]# takes randomly datas with size 1000 out of the data frame
    y_small=y[idx]
    return tx_small,y_small
