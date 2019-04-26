import glob
from bs4 import BeautifulSoup
import sys
from collections import deque
import nltk

# first few functions are just copied from earlier, to avoid imports
# they are for submission only, remove when done and import from your project instead
stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
ps = nltk.stem.PorterStemmer()


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


def build_spimi_index(path, results_path="", n_lines=100, block_size_limit_MB=0.2):
    """
    Builds spimi index - parses .sgm files in the path and handles articles one by one,
    collecting a list of term-doc_id pairs (tuples).
    Before processing next article check if the size of the list >= block_size_limit_MB,
    if yes, then call invert_and_write_to_disk method, which will create index for this block
    and save it to intermediate file.
    When all files finished, call merge_blocks function, to merge all block indexes together.
    Call the resulting file "spimi_index.dat", and intermediate block files "spimi_block_n.dat"

    :param path: path to directory with original reuters files
    :param results_path: where to save the results (spimi index as well as blocks), if not stated, saves in current dir
    :param n_lines: how many lines per block to read simultaneously when merging
    :param block_size_limit_MB: threshold for in-memory size of the term-doc_id pairs list
    :return:
    """
    # TODO write your code here


def invert_and_write_to_disk(term2doc, results_path, block_num):
    """
    Takes as an input a list of term-doc_id pairs, creates an inverted index out of them,
    sorts alphabetically by terms to allow merge and saves to a block file.
    Each line represents a term and postings list, e.g. abolish 256 1 278 2 295 2
    I.e. term doc_id_1 term_freq_1 doc_id_2 term_freq_2 ...
    See how the file should look like in the test folder
    :param term2doc: list of term-doc_id pairs
    :param results_path: where to save block files
    :param block_num: block number to use for naming a file - 'spimi_block_n.dat', use block_num for 'n'
    """
    # TODO write your code here


def merge_blocks(results_path, n_lines):
    """
    This method should merge the intermediate block files.
    First, we open all block files.
    Remember, we are limited in memory consumption,
    so we can simultaneously load only max n_lines from each block file contained in results_path folder.
    Then we find the "smallest" word, and merge all its postings across the blocks.
    Terms are sorted alphabetically, so, it allows to load only small portions of each file.
    When postings for a term are merged, we write it to resulting index file, and go to the next smallest term.
    Don't forget to sort postings by doc_id.
    As necessary, we refill lines for blocks.
    Call the resulting file 'spimi_index.dat'
    See how the file should look like in the test folder
    :param results_path: where to save the resulting index
    :param n_lines: how many lines per block to read simultaneously when merging
    """
    # TODO write your code here


def main():
    reuters_orig_path = 'your_path/to/reuters_data'
    results_path = 'results/'
    n_lines = 500
    build_spimi_index(reuters_orig_path, results_path, n_lines)


if __name__ == '__main__':
    main()