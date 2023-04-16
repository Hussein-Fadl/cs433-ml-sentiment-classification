from Glove import GloveDataset, GloveModel
import torch
import pandas as pd
import csv
import numpy as np
import string
import pickle
import torch.nn as nn
from tqdm import tqdm 
import bcolz

dataset_loaded = False

def preprocess(tweet):
    """""""""
    Preprocesses the incoming sentence as follows: 
        1)  Trimming  of  each  tweet  by  removing  leading  and trailing spaces.
        2)  Replacing all uppercase letters by lowercase ones.
        3)  Removing  punctuation,  non-alphanumeric  digits,  and unnecessary white spaces.
        4)  Constructing a vocabulary (i.e. a dictionary) of unique words in the dataset.
        5)  Tokenizing each tweet to a set of words or tokens 

    Input:
      - tweet:  an input string to be preprocessed. 
     

    Output:
      - words:  a list of tokens of the input string. 
    """""""""
    # Remove leading and trailing white space, and convert the tweet to lowercase. 
    tweet = tweet.strip().lower()
    # Remove any punctuation and numbers from the  
    #tweet = "".join([char for char in tweet if char not in string.punctuation and not char.isdigit()])
    # Tokenize the tweet 
    words = tweet.split()
    return words

def load_dataset(): 
     """""""""
    Loads the datasets and returns an array of sequences and labels.

    Output:
      - X:  a numpy array containing sequences of words for each tweet in the dataset. 
      - Y:  a numpy array containing labels (semantic) for each tweet in the dataset.
    """""""""
    X = []
    Y = []
    with open('twitter-datasets/train_pos_full.txt') as f:
            for idx, line in enumerate(f):
                X.append(preprocess(line))
                Y.append(1)
    with open('twitter-datasets/train_neg_full.txt') as f:
            for idx, line in enumerate(f):
                X.append(preprocess(line))
                Y.append(-1)

    X = np.asarray(X, dtype=object)
    Y = np.asarray(Y)
    return X, Y

def shuffle_dataset(X, Y, validation_split_ratio=0.95): 
    """""""""
    Shuffles the dataset and returns a training, and validation sets based on a validation_split_ratio
    
    Input:
      - X:  a numpy array containing sequences of words for each tweet in the dataset. 
      - Y:  a numpy array containing labels (semantic) for each tweet in the dataset.
      - validation_split_ratio: an integer representing the proportion of training set to validation set. 

    Output:
      - xtrain:  a numpy array containing training sequences.  
      - ytrain:  a numpy array containing training labels. 
      - xval:    a numpy array containing validation sequences. 
      - yval:    a numpy array containing validation labels. 
    """""""""
    np.random.seed(0)
    permutation = np.random.permutation(len(X))

    #Shuffle the data 
    X = X[permutation]
    Y = Y[permutation]

    #Split the data into training and validation set
    #split_index = int(validation_split_ratio * X.shape[0])

    xtrain = X[:-10000]
    xval = X[-10000:]
    ytrain = Y[:-10000]
    yval = Y[-10000:]
    
    return xtrain, ytrain, xval, yval

def create_embeddings(xtrain, ytrain, xval, yval, EMBED_DIM=300): 
     """""""""
    Generates the Glove embeddings for the training, and validation sets. 
    
    Input:
      - xtrain:    a numpy array containing training sequences.  
      - ytrain:    a numpy array containing training labels. 
      - xval:      a numpy array containing validation sequences. 
      - yval:      a numpy array containing validation labels.  
      - EMBED_DIM: an integer representing the dimension of the embedding space.  

    Output:
      - training_features:  a numpy array containing training sequences.  
      - validation_features:    a numpy array containing validation sequences.
      - yt:  a numpy array containing training labels.  
      - yv:    a numpy array containing validation labels. 
    """""""""
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    glove_path = "glove"
    vectors = bcolz.open(f'{glove_path}/6B.300.txt')[:]
    words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    
    matrix_len = len(vocab)
    embeddings = np.zeros((matrix_len, EMBED_DIM), dtype=float)
    words_found = 0

    for i, word in enumerate(vocab):
        try: 
            embeddings[i] = glove[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(EMBED_DIM, ))
    
    embeddings = embeddings.astype(float)

    training_features = []
    yt = []
    for i in tqdm(range(xtrain.shape[0])):
        embedding_i = 0
        count = 0
        for word in xtrain[i]:
            if word in vocab:
                embedding_i += embeddings[vocab[word]]
                count += 1
        if count > 0: 
            training_features.append(embedding_i/count)
            yt.append(ytrain[i])

    validation_features = []
    yv = []
    for i in tqdm(range(xval.shape[0])):
        embedding_i = 0
        count = 0
        for word in xval[i]:
            if word in vocab:
                embedding_i += embeddings[vocab[word]]
                count += 1
        if count > 0: 
            validation_features.append(embedding_i/count)
            yv.append(yval[i])
    
    training_features = np.asarray(training_features)
    validation_features = np.asarray(validation_features)
    yt = np.asarray(yt)
    yv = np.asarray(yv)
   
    with open('training_features.pkl', 'wb') as f:
        pickle.dump([training_features, yt], f)
        
    with open('validation_features.pkl', 'wb') as f:
        pickle.dump([validation_features, yv], f)

    return training_features, yt, validation_features, yv

def get_accuracy(model, yval): 
    """""""""
    Returns the accuracy of the model on the validation set.  
    
    Input:
      - xval:      a numpy array containing validation sequences. 
      - yval:      a numpy array containing validation labels.   

    Output:
      - accuracy:  an integer representing the accuracy of the model
    """""""""
    predictions = model(torch.from_numpy(validation_features).float().cuda()).cpu().data.numpy()

    predictions = predictions.squeeze()
    predictions[predictions<=0.5]=-1
    predictions[predictions>0.5]=1

    accuracy = np.mean(predictions==yval)*100
    return accuracy

if not dataset_loaded:
    X, Y = load_dataset()
    xtrain, ytrain, xval, yval = shuffle_dataset(X, Y) 
    print("The size of the training set is {:d}".format(xtrain.shape[0]))
    print("The size of the validation set is {:d}".format(xval.shape[0]))
    training_features, ytrain, validation_features, yval = create_embeddings(xtrain, ytrain, xval, yval, EMBED_DIM=300)
    print("The size of the training embeddings is {:d}".format(training_features.shape[0]))
    print("The size of the validation set embeddings {:d}".format(validation_features.shape[0]))
else: 
    print("Loading Training Embeddings")
    with open('training_features.pkl', 'rb') as f:
        training_features, ytrain = pickle.load(f)
    
    print("Loading Validation Embeddings")
    with open('validation_features.pkl', 'rb') as f:
        validation_features, yval = pickle.load(f)
    
class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(MLP, self).__init__()
       
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    
    def forward(self, x):
       
        return self.linear_relu_stack(x)
    
# TODO: Play with learning rate, hidden size, and optimizer type for multi-layered perceptron
learning_rate = 0.01
num_epoch = 2000

model = MLP(embedding_dim = 300, hidden_dim=200).cuda()
lossfunc = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x = torch.from_numpy(training_features).cuda()
ytrain_zero = np.copy(ytrain)
ytrain_zero[ytrain_zero==-1]=0 
print(ytrain_zero.shape)
y = torch.from_numpy(ytrain_zero).unsqueeze(1).cuda()
print(y.shape)

for epoch in range(num_epoch):

    predictions = model(x.float())
    loss = lossfunc(predictions, y.float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print ('Epoch [%d/%d], Loss: %.7f, Accuracy: %.4f' %(epoch+1, num_epoch, loss.item(), get_accuracy(model, yval)))


predictions = model(torch.from_numpy(validation_features).float().cuda()).cpu().data.numpy()

predictions = predictions.squeeze()
predictions[predictions<=0.5]=-1
predictions[predictions>0.5]=1

accuracy = np.mean(predictions==yval)*100
print("Neural Network Accuracy: {:2}%".format(accuracy))