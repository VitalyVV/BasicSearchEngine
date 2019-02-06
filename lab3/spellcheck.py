import re
import nltk
import pickle
import os
import numpy as np
from preprocess import preprocess

def build_dictionary(documents):
    """
    Build dictionary of original word forms (without stemming, but tokenized, lowercased, and only apt words considered)
    :param documents: dict of documents (contents)
    :return: {'word1': freq_word1, 'word2': freq_word2, ...}

    """
    res = dict()
    for doc_id in documents:
        ll = preprocess(documents[doc_id])
        ll = np.array(ll)
        terms, frequencies = np.unique(ll, return_counts=True)
        res.update(dict(zip(terms, frequencies)))

    return res


def build_k_gram_index(dictionary, k):
    """
    Build index of k-grams for dictionary words. Padd with '$' ($word$) before splitting to k-grams
    :param dictionary: dictionary of original words
    :param k: number of symbols in one gram
    :return: {'gram1': ['word1_with_gram1', 'word2_with_gram1', ...],
              'gram2': ['word1_with_gram2', 'word2_with_gram2', ...], ...}
    """
    words = np.array(dictionary.keys())
    words = words[:]
    res = dict()




def generate_wildcard_options(wildcard, k_gram_index):
    """
    For a given wildcard return all words matching it using k-grams
    Refer to book chapter 3.2.2
    Don't forget to pad wildcard with '$', when appropriate
    :param wildcard: query word in a form of a wildcard
    :param k_gram_index:
    :return: list of options (matching words)
    """
    # TODO write your code here


def produce_soundex_code(word):
    """
    Implement soundex algorithm, version from book chapter 3.4
    :param word: word in lowercase
    :return: soundex 4-character code, like 'k450'
    """
    # TODO write your code here


def build_soundex_index(dictionary):
    """
    Build soundex index for dictionary words.
    :param dictionary: dictionary of original words
    :return: {'code1': ['word1_with_code1', 'word2_with_code1', ...],
              'code2': ['word1_with_code2', 'word2_with_code2', ...], ...}
    """
    # TODO write your code here


print("sup")
with open('../reuters_documents.p', 'rb') as fp:
    documents = pickle.load(fp)
    build_dictionary(documents)
