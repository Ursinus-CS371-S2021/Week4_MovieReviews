import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import LogisticRegressionCV
import glob

SPECIAL_CHARACTERS = [".", "\"", "(", ")", ",", "?", "!", "\n", ":", ";", "--"]

def load_file(path):
    fin = open(path, "r")
    text = fin.read()
    # Replace the special characters in this file
    for sc in SPECIAL_CHARACTERS:
        text = text.replace(sc, "")
    return text

def load_corpus(foldername):
    # List all of the files in this directory
    paths = glob.glob("{}/*.txt".format(foldername))
    files = []
    # Loop through each file, open it up, and read it
    for p in paths:
        text = load_file(p)
        files.append(text)
    return files

##### Part 0: Load in all of the data
# Create a list of strings, where each string is a document 
pos = load_corpus("MovieReviews/pos")
neg = load_corpus("MovieReviews/neg")
all_reviews = pos + neg # First 1000 are positive, second 1000 are negative
# Label positive reviews as 1 and negative reviews as 0
labels = np.zeros(len(all_reviews))
labels[0:len(pos)] = 1 

##### Part 1: BBOW Representation

## Step 1: Create a list of lowercase words and a lookup table
## to figure out the index of a word in that list



## Step 2: Create a sparse matrix with a row for each
## review and a column for every word, and put a 1
## in every column where a word exists in that review




##### Part 2: Classification Model And Telltale Words

## Step 1: Run logistic regression


## Step 2: Print out positive/negative words




##### Part 3: Reviews of your choice
