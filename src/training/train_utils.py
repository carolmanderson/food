from collections import defaultdict
import copy
import random
import time
from typing import List

import numpy as np
from sklearn.metrics import classification_report


def get_current_time():
    t = time.localtime()
    current_time = time.strftime("%H_%M_%S", t)
    return current_time


def make_sentence_length_dict(sentence_lengths: List[int]):
    """
    Given a list of integers, each of which is the number of tokens in the document
    at that index in a dataset,  make a dict with number of tokens as key,
    list of sentence indices as value, for purposes of bucketing imilar length documents together.

    :param sentence_lengths: list of integers, each representing the number of tokens in the document
    at the corresponding index in a dataset
    :return: dictionary with integer as key, list of integers as value
    """
    sentence_length_dict = defaultdict(list)

    for i, length in enumerate(sentence_lengths):
        sentence_length_dict[length].append(i)
    return sentence_length_dict



def generate_batch_indices(batch_size: int, sentence_lengths: List[int], random_seed: int = None):
    """
    Given a set of training examples of different sizes, create minibatches that group together
    examples of similar length (to minimize the need for padding).

    :param batch_size: int, desired number of documents per batch
    :param sentence_lengths: list of ints,each of which is the number of tokens in the document
        at that index in a dataset
    :param random_seed: integer, used to set random seed for unit test
    :return: a generator that yields a list of integer indices of the documents to be used in each batch
    """


    def add_sentences_to_batch(batch_indices, batch_size, sentence_lengths, sentence_length_dict, current_index):
        num_sentences_needed = batch_size - len(batch_indices)
        current_length = sentence_lengths[current_index]
        num_sentences_available = len(sentence_length_dict[current_length])
        if num_sentences_available <= num_sentences_needed:
            batch_indices.extend(sentence_length_dict.pop(current_length))
            sentence_lengths.remove(current_length)
        else:
            for j in range(num_sentences_needed):
                chosen = random.choice(sentence_length_dict[current_length])
                batch_indices.extend([chosen])
                sentence_length_dict[current_length].remove(chosen)


    def pick_direction(sorted_list, current_length, index):
        if index >= len(sorted_list) - 1:
            return "down"
        elif index == 0:
            return "up"
        else:
            upper_diff = sorted_list[index] - current_length
            lower_diff = current_length - sorted_list[index - 1]
            if lower_diff > upper_diff:
                return "up"
            else:
                return "down"

    if random_seed:
        random.seed(random_seed)
    sentence_length_dict_original = make_sentence_length_dict((sentence_lengths))
    sentence_length_dict = copy.deepcopy(sentence_length_dict_original)
    sentence_lengths = list(sentence_length_dict.keys())
    sentence_lengths.sort()

    while sentence_lengths:
        batch_indices = []
        while len(batch_indices) < batch_size and sentence_lengths:
            current_index = random.randint(0, len(sentence_lengths) - 1)
            current_length = sentence_lengths[current_index]
            # add sentences of the same length as the first one chosen
            add_sentences_to_batch(batch_indices, batch_size, sentence_lengths, sentence_length_dict, current_index)
            # pick a direction to move (up or down)
            direction = pick_direction(sentence_lengths, current_length, current_index)
            if direction == "down":
                # if batch still isn't full, check whether smaller sentences are available
                while current_index > 0 and len(batch_indices) < batch_size:
                    current_index = current_index - 1
                    add_sentences_to_batch(batch_indices, batch_size, sentence_lengths, sentence_length_dict,
                                           current_index)

                # if batch still isn't full and if smaller sentences are exhausted, check larger ones
                while current_index < len(sentence_lengths) - 1 and len(batch_indices) < batch_size:
                    add_sentences_to_batch(batch_indices, batch_size, sentence_lengths, sentence_length_dict,
                                           current_index)

            elif direction == "up":
                while current_index < len(sentence_lengths) - 1 and len(batch_indices) < batch_size:
                    add_sentences_to_batch(batch_indices, batch_size, sentence_lengths, sentence_length_dict,
                                           current_index)

                # if batch still isn't full, check whether smaller sentences are available
                while len(sentence_lengths) > 1 and len(batch_indices) < batch_size:
                    current_index = current_index - 1
                    add_sentences_to_batch(batch_indices, batch_size, sentence_lengths, sentence_length_dict,
                                           current_index)

        yield (batch_indices)


def form_ner_train_matrices(sentence):
    """
    Form a training matrix from a single document.
    """
    tokens = np.expand_dims(sentence['tokens'], axis=0)
    labels = sentence['labels']
    labels = np.expand_dims(labels, axis=0)
    labels = np.expand_dims(labels, axis=-1)
    return np.array(tokens), np.array(labels)


def form_ner_batch_matrices(docs, token_to_index, label_to_index):
    """
    Pad batches of documents for NER model.
    """
    max_doc_length = max(len(doc['tokens']) for doc in docs)
    padding_token = token_to_index["PADDING"]
    padding_label = label_to_index["O"]
    tokens = []
    labels = []
    for doc in docs:
        padding_amount = max_doc_length - len(doc['tokens'])
        assert padding_amount >= 0
        tokens.append(doc['tokens'] + [padding_token]*padding_amount)
        labels.append(doc['labels'] + [padding_label]*padding_amount)
    return np.array(tokens), np.array(labels)


def form_ner_pred_matrix(tokens):
    tokens = np.expand_dims(tokens, axis=0)
    return np.array(tokens)


def perf_measure(y_actual, y_pred):
    """
    From https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    """
    class_id = list(set(y_actual).union(set(y_pred)))
    class_id.sort()
    TP = []
    FP = []
    TN = []
    FN = []

    for index ,_id in enumerate(class_id):
        TP.append(0)
        FP.append(0)
        TN.append(0)
        FN.append(0)
        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == _id:
                TP[index] += 1
            if y_pred[i] == _id and y_actual[i] != y_pred[i]:
                FP[index] += 1
            if y_actual[i] == y_pred[i] != _id:
                TN[index] += 1
            if y_pred[i] != _id and y_actual[i] != y_pred[i]:
                FN[index] += 1


    return class_id,    TP, FP, TN, FN


def evaluate_ner(model, sentences, index_to_label):
    label_mappings = list(index_to_label.items())
    label_mappings.sort()
    label_strings = [x[1] for x in label_mappings]
    y_pred = []
    y_true = []
    for sent in sentences:
        preds = model.predict_on_batch(form_ner_pred_matrix(sent['tokens']))
        y_pred.extend(np.argmax(preds, axis=-1)[0])
        y_true.extend(sent['labels'])
    metrics = classification_report(y_true, y_pred, target_names = label_strings,
                                    output_dict=True)
    return metrics


def pretty_print_metrics(metrics):
    for k in metrics.keys():
        print(k)
        if k == "accuracy":
            continue
        for metric in metrics[k]:
             print(metric.strip(), metrics[k][metric])


if __name__ == "__main__":
    sentence_lengths = [9, 2, 2, 30, 31, 33, 25, 40, 28, 37, 27, 1, 9, 26, 30, 35, 39, 34, 15, 16, 10, 2, 35, 35, 32, 30, 33]

    batch_size = 4

    all_batches = []
    batch_generator = generate_batch_indices(batch_size, sentence_lengths, random_seed=2)
    for batch in batch_generator:
        print("NEW BATCH:", batch)
        all_batches.append(batch)
    print(all_batches)
    #
    # check that all sentence indices got assigned to batches
    all_indices_in_batches = []
    for batch in all_batches:
        for item in batch:
            all_indices_in_batches.append(item)
    all_indices_in_batches.sort()

    expected_indices = []
    for key in sentence_length_dict:
        expected_indices.extend(sentence_length_dict[key])
    expected_indices.sort()

    assert all_indices_in_batches == expected_indices


