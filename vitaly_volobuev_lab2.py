from jellyfish import levenshtein_distance as dist
from nltk.corpus import stopwords
from collections import Counter
from nltk.collocations import *
from bs4 import BeautifulSoup
from tqdm import tqdm
import pickle
import heapq
import time
import math
import glob
import nltk
import sys
import re
import os


stop_words = set(stopwords.words('english'))


# tokenize text using nltk lib
def tokenize(text):
    return nltk.word_tokenize(text)


# checks if word is appropriate - not a stop word and isalpha
def is_apt_word(word):
    return word not in stop_words and word.isalpha()


# combines all previous methods together
def preprocess(text):
    tokenized = tokenize(text.lower())
    return [w for w in tokenized if is_apt_word(w)]


def check_on_dists(word, dictionary):
    dists = []
    for key in dictionary.keys():
        dists.append( (key, dist(key, word)) )
    mins = min(dists, key=lambda x: x[1])[:5]
    return mins

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
        intersection = [t for t in article_terms if is_apt_word(t)]
        for term in intersection:  # highlight terms for visual evaluation
            article = re.sub(r'(' + term + ')', r'\033[1m\033[91m\1\033[0m', article, flags=re.I)
        print("-------------------------------------------------------")
        print(article)

    return top_k_ids

def build_dictionary(index):
    """
    Build dictionary of original word forms (without stemming, but tokenized, lowercased, and only apt words considered)
    :param index - term:[term_frequency, (doc_id_1, doc_freq_1), (doc_id_2, doc_freq_2), ...]
    :return: {'word1': freq_word1, 'word2': freq_word2, ...}
    """
    res = dict([(x[0], x[1][0]) for x in index.items()])
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


def check_wildcard(query, index):
    dictionary = build_dictionary(index)
    k_gram = build_k_gram_index(dictionary, 3)
    options = generate_wildcard_options(query, k_gram)
    ops = check_on_dists(query, dictionary)
    options = [o for o in options if o in ops]
    print('Did you mean:')
    print("\t\n".join(options), '?')


def check_soundex(query, index):
    dictionary = build_dictionary(index)
    soundex = build_soundex_index(dictionary)
    sword = produce_soundex_code(query)
    ops = check_on_dists(query, dictionary)
    options = [o for o in soundex[sword] if o in ops]
    print('Did you mean:')
    print('\t\n'.join(options), '?')


def find_ngrams_PMI(tokenized_text, freq_thresh, pmi_thresh, n):
    """
    Finds n-grams in tokenized text, limiting by frequency and pmi value
    :param tokenized_text: list of tokens
    :param freq_thresh: number, only consider ngrams more frequent than this threshold
    :param pmi_thresh: number, only consider ngrams that have pmi value greater than this threshold
    :param n: length of ngrams to consider, can be 2 or 3
    :return: set of ngrams tuples - {('ngram1_1', 'ngram1_2'), ('ngram2_1', 'ngram2_2'), ... }
    """
    if n == 2:
        measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokenized_text)
    else:
        measures = nltk.collocations.TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(tokenized_text)

    finder.apply_freq_filter(freq_thresh)
    res = [x[0] for x in finder.score_ngrams(measures.pmi)if x[1]>pmi_thresh]

    return set(res)





def build_ngram_index(tokenized_documents, ngrams):
    """
    Builds index based on ngrams and collection of tokenized docs
    :param tokenized_documents: {doc1_id: ['token1', 'token2', ...], doc2_id: ['token1', 'token2', ...]}
    :param ngrams: set of ngrams tuples - {('ngram1_1', 'ngram1_2'), ('ngram2_1', 'ngram2_2', 'ngram2_3'), ... }
    :return: dictionary - {ngram_tuple :[ngram_tuple_frequency, (doc_id_1, doc_freq_1), (doc_id_2, doc_freq_2), ...], ...}
    """
    res = dict()
    for gramm in ngrams:
        n = len(gramm)
        for pair in tokenized_documents.items():
            doc = pair[0]
            ngram = nltk.ngrams(pair[1], n)
            gramms = []
            for gramm in ngram:
                gramms.append(gramm)

            terms = []
            freq = []
            for gramm in ngrams:
                f = gramms.count(gramm)
                if f > 0:
                    terms.append(gramm)
                    freq.append(f)

            for item in zip(terms, freq):
                it = tuple(item[0])
                if it in res:
                    if (doc, item[1]) not in res[it]:
                        res[it][0] += item[1]
                        res[it].append( (doc, item[1]) )
                else:
                    res[item[0]] = [item[1], (doc, item[1])]


    return res

#
def extract_phrases(query, documents, index, doc_lengths):
    # tokenized_documents = dict()
    # n_gramms = set()
    # for doc in tqdm(list(documents.items())[:100]):
    #     tok_doc = nltk.word_tokenize(doc[1])
    #     n_gramms |= find_ngrams_PMI(tok_doc, 2, 6, 2)
    #     tokenized_documents[doc[0]] = tok_doc

    bb = pickle.load(open('2_gramm_phrases_index.p', 'rb'))# build_ngram_index(tokenized_documents, n_gramms)
    query = query.split(' ')
    q = []
    for i in range(1, len(query)):
        q.append((query[i-1], query[i]))

    doc_ids = []
    for item in q:
        if item in bb:
            doc_ids.append( (bb[item][0], [x[0] for x in bb[item][1:]]) )
    doc_ids = sorted(doc_ids, key=lambda x: x[0])
    doc_ids = [x[1] for x in doc_ids]
    for item in doc_ids:
        for article in item:
            print("-------------------------------------------------------")
            print(documents[article])
    # pickle.dump(bb, open('2_gramm_phrases_index.p', 'wb'))
    # answer_query("apple personal computer", index, doc_lengths, documents, 5, cosine_scoring)


def build_high_low_index(index, freq_thresh):
    """
    Build high-low index based on standard inverted index.
    Based on the frequency threshold, for each term doc_ids are are either put into "high list" -
    if term frequency in it is >= freq_thresh, or in "low list", otherwise.
    high_low_index should return a python dictionary, with terms as keys.
    The structure is different from that of standard index - for each term
    there is a list - [high_dict, low_dict, len(high_dict) + len(low_dict)],
    the latter is document frequency of a term. high_dict, as well as low_dict,
    are python dictionaries, with entries of the form doc_id : term_frequency
    :param index: inverted index
    :param freq_thresh: threshold on term frequency
    :return: dictionary
    """
    terms = dict()
    for term in index:
        highs = dict()
        lows = dict()
        pairs = index[term][1:]

        for p in pairs:
            if p[1] >= freq_thresh:
                highs[p[0]] = p[1]
            else:
                lows[p[0]] = p[1]
        terms[term] = [highs, lows, len(highs) + len(lows)]

    # print(terms)
    return terms

def filter_docs(query, high_low_index, min_n_docs):
    """
    Return a set of documents in which query terms are found.
    You are interested in getting the best documents for a query, therefore you
    will sequentially check for the following conditions and stop whenever you meet one.
    For each condition also check if number of documents is  >= min_n_docs.
    1) We consider only high lists for the query terms and return a set of documents such that each document contains
    ALL query terms.
    2) We search in both high and low lists, but still require that each returned document should contain ALL query terms.
    3) We consider only high lists for the query terms and return a set of documents such that each document contains
    AT LEAST ONE query term. Actually, a union of high sets.
    4) At this stage we are fine with both high and low lists, return a set of documents such that each of them contains
    AT LEAST ONE query term.

    :param query: dictionary term:count
    :param high_low_index: high-low index you built before
    :param min_n_docs: minimum number of documents we want to receive
    :return: set if doc_ids
    """
    docs = []
    docs_low = []
    for word in query:
        high_docs = []
        low_docs = []
        if word in high_low_index:
            for d in high_low_index[word][0]:
                high_docs.append(d)
        if word in high_low_index:
            for d in high_low_index[word][1]:
                low_docs .append(d)

        docs.append(high_docs)
        docs_low.append(low_docs )

    hdoc_set = set(docs[0])
    hset = set(docs[0])

    for term in docs[1:]:
        hdoc_set = hdoc_set & set(term)
        hset = hset | set(term)

    if len(hdoc_set) >= min_n_docs:
        return hdoc_set

    if len(hset) >= min_n_docs:
        return hset

    ldoc_set = set(docs_low[0])
    anyset = set(docs_low[0])
    for term in docs_low[1:]:
        ldoc_set = ldoc_set & set(term)
        anyset = anyset | set(term)

    hlset = ldoc_set | hdoc_set
    anyset = anyset | hset

    if len(hlset) >= min_n_docs:
        return hlset

    if len(anyset) >= min_n_docs:
        return anyset


def cosine_scoring_docs(query, doc_ids, doc_lengths, high_low_index):
    """
    Change cosine_scoring function you built in the second lab
    such that you only score set of doc_ids you get as a parameter,
    and using high_low_index instead of standard inverted index
    :param query: dictionary term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built before
    :return: dictionary of scores, doc_id:score
    """
    scores = dict()

    for term in query:
        if term in high_low_index:
            qtf = math.log10(len(doc_lengths) / high_low_index[term][2])
            qw = query[term] * qtf

            for doc_id in doc_ids:
                if doc_id in high_low_index[term][0]:
                    tf_d = high_low_index[term][0][doc_id]
                else:
                    if doc_id in high_low_index[term][1]:
                        tf_d = high_low_index[term][1][doc_id]
                    else:
                        tf_d = 0
                tf_idf_d = tf_d * qtf

                if doc_id in scores.keys():
                    scores[doc_id] += qw * tf_idf_d
                else:
                    scores[doc_id] = qw * tf_idf_d

    for d in doc_lengths:
        if d in scores:
            scores[d] = scores[d] / doc_lengths[d]
        else:
            scores[d] = 0

    return scores


def okapi_scoring_docs(query, doc_ids, doc_lengths, high_low_index, k1=1.2, b=0.75):
    """
    Change okapi_scoring function you built in the second lab
    such that you only score set of doc_ids you get as a parameter,
    and using high_low_index instead of standard inverted index
    :param query: dictionary term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built before
    :return: dictionary of scores, doc_id:score
    """
    scores = dict()
    N = len(doc_lengths)
    avg_dl = sum(doc_lengths.values()) / len(doc_lengths)
    for term in query:
        if term in high_low_index:
            df = high_low_index[term][2]
            qtf = math.log(N / df) / math.log(10)
            for doc_id in doc_ids:
                if doc_id in high_low_index[term][0]:
                    tf = high_low_index[term][0][doc_id]
                else:
                    tf = high_low_index[term][1][doc_id]

                if doc_id in scores.keys():
                    scores[doc_id] += qtf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_lengths[doc_id] / avg_dl))
                else:
                    scores[doc_id] = qtf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_lengths[doc_id] / avg_dl))

    return scores


def check_inexact(query, index, doc_lenghts, documents):
    ind = build_high_low_index(index, 10)
    query = Counter(query.split(' '))
    doc_ids = filter_docs(query, ind, 3)

    t = time.time()
    ans = cosine_scoring_docs(query, doc_ids, doc_lenghts, ind)
    print(f'HIGH SCORE INDEX PASSED IN: {time.time() - t}')
    i = 0
    for article in ans:
        print("-------------------------------------------------------")
        print(documents[article])
        i += 1
        if i >= 5: break

def extract_categories(path):
    """
    Parses .sgm files in path folder wrt categories each document belongs to.
    Returns a list of documents for each category. One document usually belongs to several categories.
    Categories are contained in special tags (<TOPICS>, <PLACES>, etc.),
    see cat_descriptions_120396.txt file for details
    :param path: original data path
    :return: dict, category:[doc_id1, doc_id2, ...]
    """
    categories = dict()
    tags = ('topics', 'places', 'people', 'orgs', 'exchanges', 'companies')
    files = [
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename))
            and os.path.splitext(filename)[1] == '.sgm'
    ]
    for file in files:
        with open(file, 'r', encoding='latin-1') as in_file:
            html_data = in_file.read()
        soup = BeautifulSoup(html_data, 'html.parser')
        # Each article is enclosed in <reuters> tags
        for doc in soup.find_all('reuters'):
            doc_id = int(doc['newid'])
            for category_list in doc.find_all(tags):
                for category in category_list.find_all('d'):
                    category_name = category.string
                    if category_name not in categories:
                        categories[category_name] = list()
                    categories[category_name].append(doc_id)
    return categories


def lm_rank_documents(query, doc_ids, doc_lengths, high_low_index, smoothing, param):
    """
    Scores each document in doc_ids using this document's language model.
    Applies smoothing. Looks up term frequencies in high_low_index
    :param query: dict, term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built last lab
    :param smoothing: which smoothing to apply, either 'additive' or 'jelinek-mercer'
    :param param: alpha for additive / lambda for jelinek-mercer
    :return: dictionary of scores, doc_id:score
    """
    def get_tf(term, doc_id):
        term_freq = high_low_index[term][0].get(doc_id)
        if not term_freq:
            term_freq = high_low_index[term][1].get(doc_id)
        if not term_freq:
            term_freq = 0
        return term_freq

    if smoothing == 'additive':
        alpha = param
        vocabulary_size = len(high_low_index.keys())
    elif smoothing == 'jelinek-mercer':
        lambda_ = param
        partial_clm = dict()  # Collection language model only for query terms
        for term in query.keys():
            collection_length = 0
            partial_clm[term] = 0
            for doc_id in doc_lengths:
                partial_clm[term] += get_tf(term, doc_id)
                collection_length += doc_lengths[doc_id]
            partial_clm[term] /= collection_length
    else:
        raise ValueError(f'Unknown smoothing type: "{smoothing}"')

    scores = dict()
    for doc_id in doc_ids:
        score = 1.
        for term in query.keys():
            if smoothing == 'additive':
                score *= (
                    (get_tf(term, doc_id) + alpha)
                    / (doc_lengths[doc_id] + alpha * vocabulary_size))
            elif smoothing == 'jelinek-mercer':
                score *= (
                    (1 - lambda_) * partial_clm[term]
                    + lambda_ * get_tf(term, doc_id) / doc_lengths[doc_id])
        scores[doc_id] = score

    return scores


def lm_define_categories(query, cat2docs, doc_lengths, high_low_index, smoothing, param):
    """
    Same as lm_rank_documents, but here instead of documents we score all categories
    to find out which of them the user is probably interested in. So, instead of building
    a language model for each document, we build a language model for each category -
    (category comprises all documents belonging to it)
    :param query: dict, term:count
    :param cat2docs: dict, category:[doc_id1, doc_id2, ...]
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built last lab
    :param smoothing: which smoothing to apply, either 'additive' or 'jelinek-mercer'
    :param param: alpha for additive / lambda for jelinek-mercer
    :return: dictionary of scores, category:score
    """
    def get_tf(term, doc_id):
        term_freq = high_low_index[term][0].get(doc_id)
        if not term_freq:
            term_freq = high_low_index[term][1].get(doc_id)
        if not term_freq:
            term_freq = 0
        return term_freq

    if smoothing == 'additive':
        alpha = param
        vocabulary_size = len(high_low_index.keys())
    elif smoothing == 'jelinek-mercer':
        lambda_ = param
        partial_clm = dict()  # Collection language model only for query terms
        for term in query.keys():
            collection_length = 0
            partial_clm[term] = 0
            for doc_id in doc_lengths:
                partial_clm[term] += get_tf(term, doc_id)
                collection_length += doc_lengths[doc_id]
            partial_clm[term] /= collection_length
    else:
        raise ValueError(f'Unknown smoothing type: "{smoothing}"')

    scores = dict()
    for category in cat2docs.keys():
        score = 1.
        for term in query.keys():
            term_freq = 0
            category_length = 0
            for doc_id in cat2docs[category]:
                term_freq += get_tf(term, doc_id)
                category_length += doc_lengths[doc_id]
            if category_length > 0:
                if smoothing == 'additive':
                    score *= (
                        (term_freq + alpha)
                        / (category_length + alpha * vocabulary_size))
                elif smoothing == 'jelinek-mercer':
                    score *= (
                        (1 - lambda_) * partial_clm[term]
                        + lambda_ * term_freq / category_length)
            else:
                score = 0.
        scores[category] = score

    return scores


def extract_categories_descriptions(path):
    """
    Extracts full names for categories, draft version (inaccurate).
    You can use if as a draft for incorporating LM-based scoring to your engine
    :param path: original data path
    :return: dict, category:description
    """
    category2descr = {}
    pattern = r'\((.*?)\)'
    with open(path + 'cat-descriptions_120396.txt', 'r') as f:
        for line in f:
            if re.search(pattern, line) and not (line.startswith('*') or line.startswith('@')):
                category = re.search(pattern, line).group(1)
                if len(category.split()) == 1:
                    category2descr[category.lower()] = line.split('(')[0].strip()
    return category2descr


def check_language_model(path, query, index, doc_lengths, documents):
    # cats = extract_categories(path)
    ind = build_high_low_index(index, 3)
    query = query.split(' ')
    doc_ids = filter_docs(query, ind, 3)
    query = Counter(query)
    ans = lm_rank_documents(query, doc_ids, doc_lengths, ind, 'additive', 0.1)
    i = 0
    for article in ans:
        print("-------------------------------------------------------")
        print(documents[article])
        i += 1
        if i >= 5: break

def summarization_methods(docs, query):
    def jaccard(x, y):
        return len(set(x) & set(y)) / len(set(x) | set(y))

    for document in docs:
        sentence_list = nltk.sent_tokenize(document)

        # Calculate frequencies
        word_frequencies = dict()
        for word in nltk.word_tokenize(document):
            if word not in stop_words:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        query_frequencies = dict()
        for word in nltk.word_tokenize(query):
            if word not in stop_words:
                if word not in query_frequencies.keys():
                    query_frequencies[word] = 1
                else:
                    query_frequencies[word] += 1

        maximum_frequncy = max(word_frequencies.values())

        # Normalize
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

        # score each sentence to query
        sentence_scores = dict()
        for sent in sentence_list:
            sim = jaccard(sent.split(), query_frequencies.keys())
            for word in nltk.word_tokenize(sent.lower()):
                if word in query_frequencies.keys():
                    quer_fr = query_frequencies[word]
                else:
                    quer_fr = 2
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word] + sim * quer_fr
                        else:
                            sentence_scores[sent] += word_frequencies[word] + sim * quer_fr

        summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)

        print('---------------------------------------------')
        print(document.split('\n')[0])
        print(summary.replace('\n', ' ').replace('.', '.\n'))


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

    # t = time.time()
    # topk = answer_query("soviet union war afghanistan", index, doc_lengths, documents, 5, cosine_scoring)
    # print(f'REGULAR QUERRY PASSED: {time.time()-t}')
    # answer_query("soviet union war afghanistan", index, doc_lengths, documents, 5, okapi_scoring)

    # check_wildcard('co*puter', index)
    # check_soundex('cumputer', index)

    # extract_phrases('executive officer', documents, None, None)

    # check_inexact('soviet union war afghanistan', index, doc_lengths, documents)

    # check_language_model(reuters_path, 'soviet union war afghanistan', index, doc_lengths, documents)

    # topk = [x[1] for x in topk]
    # docs = [d[1] for d in documents.items() if d[0] in topk]
    # summarization_methods(docs, 'soviet union war afghanistan')



if __name__ == "__main__":
    main()
