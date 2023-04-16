#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:

# cat: open the file 
# sed "s/^\s\+//g": substitue the white space at the beginning of each sentence with nothing
# sort -rn: sort based on the number of occurences and reverse the order to haveit in decreasing order
# grep -v "^[1234]\s": DO NOT keep lines starting with [1234]
cat vocab_full.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt
