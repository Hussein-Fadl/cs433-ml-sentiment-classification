# ml-project-2-sei_p_2

```bash
├──DistilBERT&RoBERTa
├── ├──Cross_validation.ipynb   #Cros validation of the models
├── ├──Run_DistilBERT_RoBERTa.ipynb   #Fintuneing of the models
├── ├──Pre-trained_models.ipynb #The pretrained models
├── ├──helpers.py               #Useful helper functions
├── ├──implementations.py       #Useful implementaions
├── data
├── ├──sample_submission.csv
├── ├──test_data.txt
├── ├──twitter-datasets.zip
├── glove
├── ├──Glove.ipynb              #First implementations used for code debugging
├── ├──GloveTrainer.py          #Trains the Glove word embeddings
├── ├──Glove_FCN.py             #Trains a FCN to predict the tweet labels
├── ├──Glove_FCN_pretrained.py  #Trains a predtained FCN to predict the tweet labels
├── ├──Glove_LSTM.py            #Adds an LSTM layer on top of the Glove embddings
├── ├──Glove_LSTM_pretrained.py #Uses a pretrained LSTM to predict the tweet labels
├── word2vec
├── ├── help_wordvec.py  #USeful functions: feature creation, tokenisation etc..
├── ├── Word2cec.ipynb   #Traians all the models, Logistic Regression, KNN, SVM, FCN
├──.gitignore
├──build_vocab.sh 
├──cooc.py
├──cut_vocab.sh
├──glove_template.py
├──pickle_vocab.py
├──requirements.txt
├──run.py
├──submission_to_AIcrowd.csv 
├──README.md
```

# Installation Instructions
- Create a virtual environemnt in a desired location by running the following command: code(virtualenv ENV_NAME)
- Direct to the virtual environment source directory.
- Activate the virtual environment 
- Install the required packages from the requirements.txt file: `pip install -r requirements.txt`
- 
## In all the model the following standard libraries were used:
-`pandas` `numpy` `pickle` `matplotlib` `scipy` `torch`

## DistilBERT&RoBERTa
To run the DistilBERT&RoBERTa part make sure you have installed the following libraries:
-`collections` `json` `bz2` `importlib` `logging` `os` `bson`
- `Transformers` :Used to load the BERT/RoBERTa/DistilBERT&RoBERTa embeddings
- `Hyperopt`     :Used to optimize the hyperparameters and cross validations   
- `Tensorflow`   :Used for to have access to tensor flow 
- `sklearn`      :Used for the cross validation     
- `tqdm`         :Used to show the fancy loading bar in for loops
- `nltk`         :Used to do natural language operations

Using the DistilBERT&RoBERTa embeddings the best model was achieved. The model can be trained by running the run.py file.

## Glove
To run the Glove part make sure you have installed the following libraries:
- `nltk`         :Used to do the tokenization’s
- `Glove`        :Used to loade the pretraind models aviable on `https://nlp.stanford.edu/projects/glove/`
- `tqdm`         :Used to show the fancy loading bar in for loops
- `csv`          :Used to create csv files
- `sklearn`      :Used to construct the classification models like Logistic regression, KNN, SVM and FCN

In the glove experiments a feature matrix was created by average pooling. In addition, sequence models like Recurrent neural Network (RNN), Gated Recurrent Units (GRU), Long Short Term Memory Networks (LSTM), and bidirectional LSTM networks were constructed on top of the word embeddings. Finally, the different classifier models like, KNN, SVM and FCN were trained. 

## Word2vec
To run the Word2Vec part make sure you have installed the following libraries:
- `gensim`       :Used to create the word2vec embeddings
- `nltk`         :Used to do the tokenization’s
- `tqdm`         :Used to show the fancy loading bar in for loops (how long the for loop takes)
- `sklearn`      :Used to construct the classification models like Logistic regression, KNN, SVM and FCN 
- `csv`          :Used to create csv files

In the wordvec try’s the feature matrix was created by Average Pooling of the words in each tweet.
The tweets were tokenized by using the TweetTokenizer of the nltk library.
Finally, Logistic regression, KNN, SVM and FCN models were trained and validated.

# Run Software
#### Get predictions
To get the predictions of the best model, the user can run the file `run.py`. However, the file where the parameters of the best model are saved is too heavy for Github, therefore we ask the user to contact us in case this is required. 
Once the file `RoBERTa_finetuned_MLP_weights.h5` is saved in the main directiory, the user can simply run the command: `python run.py` to create the `submission_to_AIcrowd.csv` of predictions for the test set.

#### Train the best model
To run the training procedure of the best mode, the user can run the notebook `Run_DistilBERT_RoBERTa.ipynb`. The hyperparameters are already set to those used forthe optimal model. 
