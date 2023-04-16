from Glove import GloveDataset, GloveModel
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import csv
import numpy as np
import string
import pickle
import torch.nn as nn
from tqdm import tqdm 
from torch.autograd import Variable
from torch import LongTensor
from torch.nn import Embedding, LSTM, Linear, Dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import bcolz

def preprocess(tweet):
    """""""""
    Preprocesses the incoming sentence as follows: 
        1)  Trimming  of  each  tweet  by  removing  leading  and trailing spaces.
        2)  Replacing all uppercase letters by lowercase ones.
        3)  Removing  punctuation,  non-alphanumeric  digits,  and unnecessary white spaces.

    Input:
      - tweet:  an input string to be preprocessed. 
     

    Output:
      - tweet:  a processed string. 
    """""""""    
    # Remove leading and trailing white space, and convert the tweet to lowercase. 
    tweet = tweet.strip().lower()
    # Remove any punctuation and numbers from the  
    tweet = "".join([char for char in tweet if char not in string.punctuation and not char.isdigit()])
    # Tokenize the tweet 
    #words = tweet.split()
    return tweet

def load_dataset():
    """""""""
    Loads the datasets and returns an array of sequences and labels.

    Output:
      - X:  a numpy array containing sequences of words for each tweet in the dataset. 
      - Y:  a numpy array containing labels (semantic) for each tweet in the dataset.
    """""""""
    X = []
    Y = []
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    with open('twitter-datasets/train_pos_full.txt') as f:
        for idx, line in enumerate(f):
            line = preprocess(line)
            empty=True
            for word in line.split(): 
                if word in vocab:
                    empty=False
                    break
            if not empty: 
                X.append(preprocess(line))
                Y.append(1)
    with open('twitter-datasets/train_neg_full.txt') as f:
        for idx, line in enumerate(f):
            line = preprocess(line)
            empty=True
            for word in line.split(): 
                if word in vocab:
                    empty=False
                    break
            if not empty: 
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
    split_index = int(validation_split_ratio * X.shape[0])

    xtrain = X[:-10000]
    xval = X[-10000:]
    ytrain = Y[:-10000]
    yval = Y[-10000:]
    
    return xtrain, ytrain, xval, yval

def create_embeddings(EMBED_DIM): 
     """""""""
    Generates the Glove embeddings for the training, and validation sets. 
    
    Input:
      - EMBED_DIM: an integer representing the dimension of the embedding space.  

    Output:
      - embeddings: a numpy array containing the embeddings for each word in the vocabulary. 
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
    return embeddings

def get_accuracy(model, xval, yval):
    """""""""
    Returns the accuracy of the model on the validation set.  
    
    Input:
      - xval:      a numpy array containing validation sequences. 
      - yval:      a numpy array containing validation labels.   

    Output:
      - accuracy:  an integer representing the accuracy of the model
    """""""""
    predictions, perm_idx = model(xval)
    predictions = predictions.cpu().data.numpy().squeeze()
    perm_idx = perm_idx.cpu().data.numpy()
    y = yval[perm_idx]
    predictions[predictions<=0]=-1
    predictions[predictions>0]=1
    return np.mean(predictions==y)*100

class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
        
    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.text[idx]
            sample = {"Text": text, "Class": label}
            return sample


class LSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, output_dim, embeddings, batch_size):
        
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = False
        self.hidden_dim = hidden_dim 
        
        #self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.fc_stack = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim))
#         self.hidden = self.init_hidden(hidden_dim, batch_size)

    
    def init_hidden(self, hidden_dim, batch_size):
        return (Variable(torch.zeros(1, batch_size, hidden_dim)),
                Variable(torch.zeros(1, batch_size, hidden_dim)))
        
    def forward(self, text):
        #text = [sent len, batch size]
        vectorized_seqs = [[vocab[tok] for tok in seq.split() if tok in vocab]for seq in text]
        seq_lengths = LongTensor(list(map(len, vectorized_seqs))).cuda()
        seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long().cuda()
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = LongTensor(seq).cuda()
        
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx].cuda()

        embedded_seq_tensor = self.embedding(seq_tensor)
        
        packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
        
        #output, hidden = self.rnn(packed_input)
        output, (hidden, cell) = self.lstm(packed_input.float())
        output, (hidden, cell) = self.lstm2(output)
        ####
        output, _ = pad_packed_sequence(output, batch_first=True)
        out_forward = output[range(len(output)), output.shape[1] - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        hidden = torch.cat((out_forward, out_reverse), 1)
#         hidden = self.dropout(hidden)
        ####
        
        return self.fc_stack(hidden.squeeze(0)), perm_idx


           
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    
num_epoch = 10
num_embeddings = len(vocab)
embedding_dim = 300 
hidden_dim = 200
output_dim = 1
lr = 5e-3
batch_size = 128

X, Y = load_dataset()
xtrain, ytrain, xval, yval = shuffle_dataset(X, Y) 
print("The size of the training set is {:d}".format(xtrain.shape[0]))
print("The size of the validation set is {:d}".format(xval.shape[0]))
embeddings = create_embeddings(embedding_dim)
    

dataset = CustomTextDataset(xtrain, ytrain)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
model = LSTM(num_embeddings, embedding_dim, hidden_dim, output_dim, embeddings, batch_size).cuda()
lossfunc = nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    for (idx, batch) in enumerate(dataloader):
        x = batch['Text']
        y = batch['Class'].cuda()
        y[y<0] = 0
        predictions, perm_index = model(x)
        loss = lossfunc(predictions.squeeze(), y[perm_index].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        print ('Epoch [%d/%d], Loss: %.7f, Accuracy: %.4f' %(epoch+1, num_epoch, loss.item(), get_accuracy(model, xval, yval)))

predictions, perm_idx = model(xval)
predictions = predictions.cpu().data.numpy().squeeze()
perm_idx = perm_idx.cpu().data.numpy()
y = yval[perm_idx]
predictions[predictions<=0]=-1
predictions[predictions>0]=1
accuracy = np.mean(predictions==y)*100
print(predictions)
print(y)
print("Neural Network Accuracy: {:2}%".format(accuracy))