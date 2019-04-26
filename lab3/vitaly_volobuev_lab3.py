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
    res = dict()
    words = list(map(lambda x: '$' + x + '$', list(dictionary.keys())))
    for item in words:
        for i in range(len(item) - k + 1):
            gram = item[i:i + k]
            if gram not in res:
                res[gram] = []
            res[gram].append(item.replace('$', ''))

    return res


def generate_wildcard_options(wildcard, k_gram_index):
    """
    For a given wildcard return all words matching it using k-grams
    Refer to book chapter 3.2.2
    Don't forget to pad wildcard with '$', when appropriate
    :param wildcard: query word in a form of a wildcard
    :param k_gram_index:
    :return: list of options (matching words)
    """
    def k_gramm_query(query, k):
        res = []
        for i in range(len(query)):
            part = query[i:min(i + k, len(query))]
            res.append(part)
            if i+k >= len(query):
                break

        return res

    s = list(k_gram_index.keys())
    k = len(s[0])
    wildcard = f'${wildcard}$'
    quer = wildcard.split('*')
    ans_set = set()
    for gram in quer:
        query = k_gramm_query(gram, k)
        for part in query:
            if part in k_gram_index:
                if len(ans_set) > 0:
                    ans_set = ans_set & set(k_gram_index[part])
                else:
                    ans_set = ans_set | set(k_gram_index[part])


    return list(ans_set)



def produce_soundex_code(word):
    """
    Implement soundex algorithm, version from book chapter 3.4
    :param word: word in lowercase
    :return: soundex 4-character code, like 'k450'
    """

    mapp = {'a': '0', 'e': '0', 'i': '0', 'o': '0', 'u': '0', 'h': '0', 'w': '0', 'y': '0',
                'b': '1', 'f': '1', 'p': '1', 'v': '1',
                'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
                'd': '3', 't': '3',
                'l': '4',
                'm': '5', 'n': '5',
                'r': '6'}

    code = word[0]
    prev = '0'

    for ch in word[1:]:
        sym = mapp[ch]
        if prev == sym:
            prev = sym
            continue

        prev = sym
        if sym!='0':
            code += sym

    res = str(code[:4])
    if len(res)<4:
        for i in range(4 - len(res)):
            res += '0'
    return res



def build_soundex_index(dictionary):
    """
    Build soundex index for dictionary words.
    :param dictionary: dictionary of original words
    :return: {'code1': ['word1_with_code1', 'word2_with_code1', ...],
              'code2': ['word1_with_code2', 'word2_with_code2', ...], ...}
    """
    res = {}
    for word in dictionary.keys():
        code = produce_soundex_code(word)
        if code not in res:
            res[code] = []
        res[code].append(word)

    return res


with open('../reuters_documents.p', 'rb') as fp:
    # documents = pickle.load(fp)
    # dictio = build_dictionary(documents)
    # print(dictio)
    # k_gram = build_k_gram_index(dictio, 3)
    soundex_test = {"britney": "b635",
                    "britain": "b635",
                    "priteny": "p635",
                    "retrieval": "r361",
                    "ritrivl": "r361",
                    "lorem": "l650",
                    "lorrrremmn": "l650",
                    "awe": "a000"}
    passed = True
    for pair in soundex_test.items():
        code = produce_soundex_code(pair[0])
        print(f'Obtained code: {code}')
        print(f'Expected pair: {pair[1]}')
        print(code == pair[1])



