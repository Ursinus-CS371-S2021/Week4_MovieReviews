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

def get_reviews():
    """
    Load in a bunch of movie reviews that are both positive and negative
    
    Returns
    -------
    all_reviews: N-length list of lists of strings
        Each movie review, represented as a list of lowercase words,
            excluding special characters.  
        For example, if there were two movie reviews: 
            "This is awesome!"
            "This movie bites"
        Then all_reviews would look like
        [ ["this", "is", "awesome"], ["this", "movie", "bites"] ]
    
    labels: ndarray(N)
        A parallel array holding whether each review is positive (1) or negative(0)
    """
    # Create a list of strings, where each string is a document 
    pos = load_corpus("MovieReviews/pos")
    neg = load_corpus("MovieReviews/neg")
    all_reviews = pos + neg # First 1000 are positive, second 1000 are negative
    # Label positive reviews as 1 and negative reviews as 0
    labels = np.zeros(len(all_reviews))
    labels[0:len(pos)] = 1 
    # Split each review into a list of lowercase strings
    for i, review in enumerate(all_reviews):
        all_reviews[i] = [s.lower() for s in review.split()]
    return all_reviews, labels


##### Part 1: BBOW Representation

## Step 1: Create a list of unique lowercase words 
## across all movie reviews, as well as a lookup table
## to fix unique indices for each word

all_reviews, labels = get_reviews()
words = set([]) # A set of all lowercase words across all reviews
word_index = {} # A map from a lowercase word to its column index

## TODO: Fill in code to setup the words set and the words_index map


## Step 2: Create a sparse matrix with a row for each
## review and a column for every word, and put a 1
## in every column where a word exists in that review
N = len(all_reviews)
d = len(words)
X = sparse.lil_matrix((N, d))

## TODO: Fill in the matrix X.  All entries are 0 by default
## but you should put a 1 in row i and column j if movie review
## i contains the word at index j



##### Part 2: Classification Model And Telltale Words

# Run logistic regression
clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, labels)
print("Classification Accuracy: {:.3f}%\n\n".format(100*clf.score(X, labels)))

# Print out positive/negative words
print("Negative: ")
words = list(words)
for idx in np.argsort(clf.coef_.flatten())[0:15]:
    print(words[idx])

print("\n\nPositive: ")
for idx in np.argsort(-clf.coef_.flatten())[0:15]:
    print(words[idx])
