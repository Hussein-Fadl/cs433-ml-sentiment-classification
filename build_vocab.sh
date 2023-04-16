#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:

# cat: opens files and concatenates them if more than one is inputed 
# sed "s/ /\n/g": substitute blank spacewith \n, for all white spaces. That is, tokenize using only white spaces 
# grep -v: DO NOT keep lines that start with white space and have white spaces everywhere
# sort: sort tokenz 
# uniq -c: keep only one instance for each token and count how many times it appeared
cat twitter-datasets/train_pos_full.txt twitter-datasets/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_full.txt
