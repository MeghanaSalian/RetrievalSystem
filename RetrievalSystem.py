import nltk
from heapq import nsmallest
from nltk.tokenize import RegexpTokenizer
import os

def CountFrequency(my_list):
    """ 
    Function to count the frequency of repeated words.

    Arguments:
    my_list: List of words in the document.

    Returns:
    count : A dictionary containing the word as key and frequency as value.
    """

    count = {}  
    for i in my_list: 
        count[i] = count.get(i, 0) + 1 
    return count

def IncrementFrequency(list_of_words, reinit_global_dict):	#
    """
    Compute frequency of words and increment to the vocabulary dictionary.
    
    Arguments:
    list_of_words: A list of words in the document.
    reinit_global_dict: Dictionary of all known vocabulary.
    
    Returns:
    reinit_global_dict: Updated dictionary with added frequencies.
    """
    
    for word in list_of_words:
        if word in reinit_global_dict:
            reinit_global_dict[word] += 1
    return reinit_global_dict

def IncrementDocument(list_of_words, term_occurs): # 
    """
    Increment term_occurs for Calculting IDF.
    
    Arguments:
    list_of_words: A list of words in the document.
    term_occurs: Dictionary holding the number of documents each word occurs in.
    
    Returns:
    term_occurs: Updated dictionary with added frequencies.
    """
    
    uniq_words = set(list_of_words)
    for word in uniq_words:
        term_occurs[word] += 1
    return term_occurs

os.chdir('Dataset-P1')
filelist = os.listdir() #list all data files in Dataset-P1 folder

print("Reading documents and constructing vocabulary")
wordlist = list()
for filename in filelist:
    with open(filename, 'r') as f:
        word = f.read()
        tokenizer = RegexpTokenizer(r'\w+')
        wordlist.extend(tokenizer.tokenize(word)) # Extract the word from the file and adds it to the list
print("Done!")

count_dict = CountFrequency(wordlist) # Counts the number of times each word is repeated in entire dataset

term_occurs = dict.fromkeys(count_dict,0 ) # Set value of all values in count_dict to 0
tf = list()

print("\nComputing TF and IDF values")
for newfilename in filelist:
    with open(newfilename, 'r') as f:
        word = f.read()
        if word:
            tokenizer = RegexpTokenizer(r'\w+')
            tokenized = tokenizer.tokenize(word)
            
            wordFrequency = dict.fromkeys(count_dict,0 )  
            incremented_dict = IncrementFrequency(tokenized, wordFrequency) # Count the value occurance
            vector = list(incremented_dict.values())
            term_occurs = IncrementDocument(tokenized, term_occurs) # Counts the number of documents for each word
            tf.append(vector) # TF values for each document is appended
        else:
            print(newfilename + " is empty. Skipping...")
print("Done!")
idf_nj = list(term_occurs.values()) #number of document in which a word has occured

# inverse document frequency
import numpy as np
n=len(filelist)
nj=np.array(idf_nj)
idf_j = np.log2(n/nj) + 1  
tf_ij = np.array(tf)


#TF-IDF values
tf_idf = tf_ij * idf_j


###################


# Euclidean distance
from scipy.spatial import distance
euc_distances = dict()
for i in range(1, len(tf_idf)):
    d = distance.euclidean(tf_idf[0], tf_idf[i]) #
    euc_distances[filelist[i]] = round(d, 2)

top10 = nsmallest(10, euc_distances, key=euc_distances.get) #computing the top 10 most popular words
print("\nTop 10 relevant documents using TF-IDF values.")
print("Document   Distance")
for document_name in top10:
    print(document_name + " - " + str(euc_distances[document_name]))


###################


#Cosine Similarity
from scipy import spatial

cos_distances = dict()
for i in range(1, len(tf)):
    csd = 1 - spatial.distance.cosine(tf[0], tf[i])
    cos_distances[filelist[i]] = round(csd, 2)

top10_cosin = nsmallest(10, cos_distances, key=cos_distances.get) #computing the top 10 most popular words
print("\nTop 10 relevant documents using Cosine distance.")
print("Document  Distance")
for document_name in top10_cosin:
    print(document_name + " - " + str(cos_distances[document_name]))
