###################################
#Jonathan Pedoeem                 #
#Natural Language Processing      #
#Project 1 - Text Categorization  #
###################################
from collections import Counter
import math
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")

def normalize_dict(dictionary):
    vals   = dictionary.values()
    factor = 1.0/np.sqrt(np.sum(np.multiply(vals,vals)))
    for term in dictionary:
        temp = dictionary[term]
        dictionary[term] = temp*factor
    return dictionary

def update_frequencies(d_dir, term_frequncy,document_frequency=False, train=False):
    document_text = [stemmer.stem(w) for w in tokenizer.tokenize(open(d_dir, "r").read().lower())]
    frequncy_count= Counter(document_text)
    term_frequncy[d_dir] = frequncy_count
    if train:
        #will update the document frequncy here:
        for word in frequncy_count:
            if word in document_frequency:
                temp = document_frequency[word]
                document_frequency[word] = temp + 1
            else:
                document_frequency[word] = 1
        return document_frequency, term_frequncy
    return term_frequncy

def tfidf(doc_term_frequency,document_frequency, num_docs,cat=False,centroid=False):
    sum_values = 0.0
    max_word_freq= max(doc_term_frequency.values())
    for term in doc_term_frequency:
        temp = 0.5 + 0.5*(doc_term_frequency[term]/max_word_freq)
        doc_freq = 0
        if term in document_frequency:
            doc_freq = math.log(num_docs/document_frequency[term])
            tfidf = temp * doc_freq
            sum_values += tfidf*tfidf
            doc_term_frequency[term] = tfidf
    factor = 1.0/math.sqrt(sum_values)
    for term in doc_term_frequency:
        temp = doc_term_frequency[term]
        normal = temp* factor
        doc_term_frequency[term] = normal
        if cat:
            if term in centroid:
                temp_centroid = centroid[term]
                centroid[term]= temp_centroid + normal/num_docs
            else:
                centroid[term]= normal/num_docs
    if cat:
        return doc_term_frequency, centroid
    return doc_term_frequency

def compare(test_vect, centroid):
    score = 0
    for term in test_vect:
        if term in centroid:
            score += test_vect[term] * centroid[term]
    return score

def train(training_file):
    training_list       = open(training_file,"r").read().split("\n")
    #document frequncy is a dictionary where the keys are words and values are the number of documents the word appears in
    document_frequency   = {}
    # document cat lookup is a dictionary where the keys are documents and the values are its categories.
    document_cat_lookup = {}
    #term frequncy is a dictionray where the key is a document directory and the value is a dictionary with term frequency
    term_frequncy       = {}
    #centroid is a dictionary where the keys are categories and the values are dictionaries of normalized centroid article_vectors
    centroids           = {}
    for document in training_list:
            #Will update the three dictionaries.
            splited       = document.split()
            if splited:
                d_dir, d_type = splited
                document_cat_lookup[d_dir] = d_type
                document_frequency, term_frequncy = update_frequencies(d_dir,term_frequncy,document_frequency=document_frequency, train=True)
    #now need to calculate tf-id of each term in each document and then create the centroid.
    num_docs = len(document_frequency)
    for document in term_frequncy:
        doc_term_frequency      = term_frequncy[document]
        category                = document_cat_lookup[document]
        if category in centroids:
            centroid                = centroids[category]
        else:
            centroid = {}
        term_frequncy[document], centroids[category] = tfidf(doc_term_frequency,document_frequency,num_docs,cat=category, centroid=centroid)
    for c in centroids:
        centroids[c] = normalize_dict(centroids[c])
    return centroids, document_frequency, num_docs


def test(testing_file, centroids,num_docs, document_frequency,output_file_name):
    test_list   = open(testing_file,"r").read().split("\n")
    results     = open(output_file_name, 'wb')
    for d_dir in test_list:
        #get the normalized tf-idf
        if d_dir:
            d_dir = d_dir.rstrip()
            word_freq   = update_frequencies(d_dir,{})
            tf_idf_freq = tfidf(word_freq[d_dir],document_frequency,num_docs)
            #do the comparison
            cosine_score =None
            category     =None
            for c in centroids:
                result = compare(tf_idf_freq,centroids[c])
                if result>cosine_score:
                    cosine_score = result
                    category     = c
        #put it in the document
        results.write("{} {}\n".format(d_dir, category))
    results.close()

if __name__ == "__main__":
    argv        = sys.argv
    centroids, document_frequency,num_docs = train(argv[1])
    output_name = raw_input("Please enter a name for the output file \n")
    test(argv[2],centroids,num_docs,document_frequency,output_name)
