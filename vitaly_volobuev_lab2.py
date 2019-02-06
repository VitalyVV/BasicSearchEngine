from collections import Counter
from bs4 import BeautifulSoup
import glob
import nltk
import pickle
import math
import heapq
import re
import os

stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
ps = nltk.stem.PorterStemmer()


# tokenize text using nltk lib
def tokenize(text):
    return nltk.word_tokenize(text)


# stem word using provided stemmer
def stem(word, stemmer):
    return stemmer.stem(word)


# checks if word is appropriate - not a stop word and isalpha
def is_apt_word(word):
    return word not in stop_words and word.isalpha()


# combines all previous methods together
def preprocess(text):
    tokenized = tokenize(text.lower())
    return [stem(w, ps) for w in tokenized if is_apt_word(w)]


def build_index(path, limit=None):
    """
    # principal function - builds an index of terms in all documents
    # generates 3 dictionaries and saves on disk as separate files:
    # index - term:[term_frequency, (doc_id_1, doc_freq_1), (doc_id_2, doc_freq_2), ...]
    # doc_lengths - doc_id:doc_length
    # documents - doc_id: doc_content_clean
    :param path: path to directory with original reuters files
    :param limit: number of articles to process, for testing. If limit is not None,
                  return index when done, without writing files to disk
    """
    f = [fl for fl in glob.glob(path + "*.sgm")]
    index = dict()
    lengths = dict()
    docs = dict()

    n_articles = 0
    for doc in f:
        reuter_stream = open(doc, encoding="latin-1")
        reuter_content = reuter_stream.read()
        soup = BeautifulSoup(reuter_content, "html.parser")
        articles = soup.find_all('reuters')

        # BEGGINING OF PARSING SINGLE ARTICLE IN DOCUMENT
        for article in articles:

            doc_id = int(article['newid'])
            n_articles += 1
            print(f'{n_articles}: Processing article id №: {doc_id}')

            titl = []
            text = ''
            if article.find('title'):
                titl = preprocess(article.title.string)
                text += article.title.string + '\n'

            body = []
            if article.find('body'):
                # APPLY PREPROCESSING
                body = preprocess(article.body.string)
                text += article.body.string

            docs[doc_id] = text

            art_words = titl + body
            art_wordfreq = [art_words.count(w) for w in art_words]
            art_termfreq = dict(set(zip(art_words, art_wordfreq))) # dict of tuples (term, frequency)

            lengths[doc_id] = len(art_words)
            for item in art_termfreq:
                if item not in index:
                    index[item] = [0]
                index[item][0] += art_termfreq[item]
                index[item].append((doc_id, art_termfreq[item]))

            if (not limit is None) and n_articles >= limit:
                break

        if ( not limit is None ) and n_articles >= limit:
            break

    print('Saving data on disk...')
    with open('reuters_index.p', 'wb+') as reuters_index:
        pickle.dump(index, reuters_index)

    with open('reuters_doc_lengths.p', 'wb+') as reuters_doc_lengths:
        pickle.dump(lengths, reuters_doc_lengths)

    with open('reuters_documents.p', 'wb+') as reuters_documents:
        pickle.dump(docs, reuters_documents)

    return index



def cosine_scoring(query, doc_lengths, index):
    """
    Computes scores for all documents containing any of query terms
    according to the COSINESCORE(q) algorithm from the book (chapter 6)

    :param query: dictionary - term:frequency
    :return: dictionary of scores - doc_id:score
    """
    scores = dict()

    for term in query:
        if term in index:
            docs = index[term][1:]
            qtf = math.log(len(index) / len(docs), 10)
            qw = query[term] * qtf
            for item in docs:
                doc_id, doc_freq = item
                dtw = doc_freq * qtf
                if doc_id not in scores.keys():
                    scores[doc_id] = 0
                scores[doc_id] += qw * dtw

    for d in doc_lengths:
        if d in scores:
            scores[d] = scores[d]/doc_lengths[d]
        else:
            scores[d] = 0

    return scores

def okapi_scoring(query, doc_lengths, index, k1=1.2, b=0.75):
    """
    Computes scores for all documents containing any of query terms
    according to the Okapi BM25 ranking function, refer to wikipedia,
    but calculate IDF as described in chapter 6, using 10 as a base of log

    :param query: dictionary - term:frequency
    :return: dictionary of scores - doc_id:score
    """
    scores = dict()
    avg_dl = sum(doc_lengths.values()) / len(doc_lengths)
    for term in query:
        if term in index:
            docs = index[term][1:]
            qtf = math.log(len(index) / len(docs), 10)
            for item in docs:
                doc_id, doc_freq = item
                okami = (doc_freq * (k1 + 1)) / \
                            (doc_freq + k1 * (1 - b + b * doc_lengths[doc_id] / avg_dl))

                if doc_id not in scores.keys():
                    scores[doc_id] = 0
                scores[doc_id] += qtf * okami

    return scores



def answer_query(raw_query, index, doc_lengths, documents, top_k, scoring_fnc):
    """
    :param raw_query: user query as it is
    :param top_k: how many results to show
    :param scoring_fnc: cosine/okapi
    :return: list of ids of retrieved documents (top_k)
    """
    # pre-process query the same way as documents
    query = preprocess(raw_query)
    for i in range(len(query)):
        query[i] = query[i].strip()

    # count frequency
    query = Counter(query)
    # retrieve all scores
    scores = scoring_fnc(query, doc_lengths, index)
    # put them in heapq data structure, to allow convenient extraction of top k elements
    h = []
    for doc_id in scores.keys():
        neg_score = -scores[doc_id]
        heapq.heappush(h, (neg_score, doc_id))
    # retrieve best matches
    top_k = min(top_k, len(h))  # handling the case when less than top k results are returned
    print('\033[1m\033[94mANSWERING TO:', raw_query, 'METHOD:', scoring_fnc.__name__, '\033[0m')
    print(top_k, "results retrieved")
    top_k_ids = []
    for k in range(top_k):
        best_so_far = heapq.heappop(h)
        top_k_ids.append(best_so_far)
        article = documents[best_so_far[1]]
        article_terms = tokenize(article)
        intersection = [t for t in article_terms if is_apt_word(t) and stem(t, ps) in query.keys()]
        for term in intersection:  # highlight terms for visual evaluation
            article = re.sub(r'(' + term + ')', r'\033[1m\033[91m\1\033[0m', article, flags=re.I)
        print("-------------------------------------------------------")
        print(article)

    return top_k_ids


def main():
    reuters_path = 'reuters21578/'
    if not os.path.isfile('reuters_index.p'):
        build_index(reuters_path)
    with open('reuters_index.p', 'rb') as fp:
        index = pickle.load(fp)
    with open('reuters_doc_lengths.p', 'rb') as fp:
        doc_lengths = pickle.load(fp)
    with open('reuters_documents.p', 'rb') as fp:
        documents = pickle.load(fp)
    # answer_query("soviet union war afghanistan", index, doc_lengths, documents, 5, cosine_scoring)
    # answer_query("soviet union war afghanistan", index, doc_lengths, documents, 5, okapi_scoring)

    # answer_query("black monday", index, doc_lengths, documents, 5, cosine_scoring)
    # answer_query("black monday", index, doc_lengths, documents, 5, okapi_scoring)

    answer_query("apple personal computer", index, doc_lengths, documents, 5, cosine_scoring)
    answer_query("apple personal computer", index, doc_lengths, documents, 5, okapi_scoring)


if __name__ == "__main__":
    main()
