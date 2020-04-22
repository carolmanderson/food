import random

import numpy as np

def read_conll_file(file):
    """
    Given a file in CoNLL format, read in tokens and labels. Treat each sentence as a training example.
    :param file: file in CoNLL format; tokens are assumed to be in the first column and labels in the last column.
    :returns a nested list, in which each sublist is a sentence and contains a sublist [token, label] for each token.

    .. note:: Ignores document boundaries and treats each sentence as an independent training example.
    """
    documents = []  # holds all documents
    sentence = [] # will hold the first sentence
    with open(file, 'r') as infile:
        for line in infile:
            if '-DOCSTART-' in line:  # beginning of a new document; ignore this since we will treat each sentence as a training example
                continue
            elif not line.split():  # beginning of a new sentence
                if sentence:
                    documents.append(sentence)
                sentence = []
            else:
                token, *other_columns, label = line.split()
                sentence.append([token, label])
    return documents


def compile_vocabulary(dataset, lowercase=False):
    """
    Given a dataset, compile a dict of tokens and their frequency counts.

    :param dataset list: a nested list produced by read_conll_file (in which each sublist is a training example and
    contains a sublist [token, label] for each token)
    :param lowercase boolean: whether to lowercase all tokens when compiling vocabulary
    :returns dictionary with all tokens in the dataset as keys, and
    their frequency counts as values.
    """
    vocabulary = {}
    for sentence in dataset:
        for item in sentence:
            token = item[0]
            # add the token to the vocabulary
            if lowercase:
                token = token.lower()
            if token in vocabulary:
                vocabulary[token] += 1
            else:
                vocabulary[token] = 1
    return vocabulary


def make_label_map(dataset):
    """
    Given a dataset, assign an index to each label and return label_to_index mappings.

    :param dataset list: a nested list produced by read_conll_file (in which each sublist is a training example and
    contains a sublist [token, label] for each token)
    :return: dictionary with label as key, index as value
    """
    label_set = set()
    for sentence in dataset:
        for item in sentence:
            label = item[1]
            label_set.add(label)
    label_to_index = {}
    for label in label_set:
        label_to_index[label] = len(label_to_index)
    return label_to_index


def get_token_embeddings(embeddings_file, embedding_dim, vocabulary, token_frequency_threshold):
    """
    Create a mapping of token to index, and also make a list of embeddings.

    :param embeddings_file string: file holding the embeddings in Glove format
    :param embedding_dim integer: dimension of the embeddings
    :param vocabulary dict: dictionary with all tokens in the dataset as keys, and
    their frequency counts as values
    :param token_frequency_threshold int: if the token appears at least this
    number of times in this dataset, make a new randomly initialized embedding for it

    :returns:
        -token_to_index: dictionary with token string as key, index as value
        -embeddings: list of numpy arrays, each of which is an embedding. The embedding's index in the
        list corresponds to its index in token_to_index.
    """

    token_to_index = {"PADDING":0, "UNKNOWN_TOKEN":1, "NUMERIC": 2}
    embeddings = []

    # add padding token to embeddings
    vector = np.zeros(embedding_dim)
    embeddings.append(vector)
    # add a random vector to initialize unknown token
    vector = np.random.uniform(-0.25, 0.25, embedding_dim)
    embeddings.append(vector)

    # iterate through embeddings and pull out the tokens found in this dataset
    with open(embeddings_file, 'r') as emb_file:
        for line in emb_file:
            token, *vector = line.rstrip().split(" ")
            assert len(vector) == embedding_dim
            if token in vocabulary:
                token_to_index[token] = len(token_to_index)
                embeddings.append(vector)

    # check for tokens found abundantly in this dataset but not in the embeddings
    # make an embedding for the lowercase version of the token
    for token in vocabulary:
        if token in token_to_index:
            continue
        if token.lower() in token_to_index:
            continue
        elif vocabulary[token] > token_frequency_threshold:  # if the token appears at least this number of times in this dataset, make an embedding for it
            vector = np.random.uniform(-0.25, 0.25, embedding_dim)
            embeddings.append(vector)
            token_to_index[token.lower()] = len(token_to_index)

    embeddings = np.array(embeddings)
    return token_to_index, embeddings


def examples_to_indices(dataset, label_to_index, token_to_index):
    """
    Given training examples with tokens and labels as strings, convert them to indices.
    :param dataset list: a nested list produced by read_conll_file (in which each sublist is a training example and
    contains a sublist [token, label] for each token)
    :param label_to_index dict: label as key, index as value
    :param token_to_index dict: token as key, index as value
    :returns list: list in which each training example is a dict with
    'tokens' and 'labels' as keys, and list of token or label indices as values.
    """
    sentence_list = []
    for sentence in dataset:
        token_indices = []
        label_indices = []
        for item in sentence:
            token = item[0]
            label = item[1]
            label_indices.append(label_to_index[label])
            if token in token_to_index:
                token_indices.append(token_to_index[token])
            elif token.lower() in token_to_index:
                token_indices.append(token_to_index[token.lower()])
            elif token.isnumeric():
                token_indices.append(token_to_index["NUMERIC"])
            else:
                token_indices.append(token_to_index["UNKNOWN_TOKEN"])
        sentence_list.append({"tokens" : token_indices, "labels": label_indices})
    return sentence_list





if __name__ == "__main__":
    datafile = "/Users/Carol/Documents/Repos/NER-datasets/CONLL2003/tiny_train.txt"
    embeddings_file = "/Users/Carol/Dropbox/Code/Glove/glove.6B.100d.txt"
    embedding_dim = 100
    token_frequency_threshold = 5
    dataset = read_conll_file(datafile)
    print(dataset)

    vocabulary = compile_vocabulary(dataset)
    print(vocabulary)

    token_to_index, embeddings = get_token_embeddings(embeddings_file, embedding_dim, vocabulary,
                                                      token_frequency_threshold)
    label_to_index = make_label_map(dataset)

    sentences = examples_to_indices(dataset, label_to_index, token_to_index)
    print(sentences)

