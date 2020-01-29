import pytest

from src.data_prep.prodigy_to_conll import *


@pytest.fixture
def tokens():
    return [{'text': 'Combine', 'start': 0, 'end': 7, 'id': 0},
            {'text': 'all', 'start': 8, 'end': 11, 'id': 1},
            {'text': 'ingredients', 'start': 12, 'end': 23, 'id': 2},
            {'text': 'in', 'start': 24, 'end': 26, 'id': 3},
            {'text': 'large', 'start': 27, 'end': 32, 'id': 4},
            {'text': 'bowl', 'start': 33, 'end': 37, 'id': 5},
            {'text': ';', 'start': 37, 'end': 38, 'id': 6},
            {'text': 'toss', 'start': 39, 'end': 43, 'id': 7},
            {'text': 'to', 'start': 44, 'end': 46, 'id': 8},
            {'text': 'blend', 'start': 47, 'end': 52, 'id': 9},
            {'text': '.', 'start': 52, 'end': 53, 'id': 10},
            {'text': 'Season', 'start': 54, 'end': 60, 'id': 11},
            {'text': 'salsa', 'start': 61, 'end': 66, 'id': 12},
            {'text': 'to', 'start': 67, 'end': 69, 'id': 13},
            {'text': 'taste', 'start': 70, 'end': 75, 'id': 14},
            {'text': 'with', 'start': 76, 'end': 80, 'id': 15},
            {'text': 'salt', 'start': 81, 'end': 85, 'id': 16},
            {'text': 'and', 'start': 86, 'end': 89, 'id': 17},
            {'text': 'pepper', 'start': 90, 'end': 96, 'id': 18},
            {'text': '.', 'start': 96, 'end': 97, 'id': 19},
            {'text': 'Cover', 'start': 98, 'end': 103, 'id': 20},
            {'text': 'salsa', 'start': 104, 'end': 109, 'id': 21},
            {'text': 'and', 'start': 110, 'end': 113, 'id': 22},
            {'text': 'refrigerate', 'start': 114, 'end': 125, 'id': 23},
            {'text': 'at', 'start': 126, 'end': 128, 'id': 24},
            {'text': 'least', 'start': 129, 'end': 134, 'id': 25},
            {'text': '2', 'start': 135, 'end': 136, 'id': 26},
            {'text': 'hours', 'start': 137, 'end': 142, 'id': 27},
            {'text': 'and', 'start': 143, 'end': 146, 'id': 28},
            {'text': 'up', 'start': 147, 'end': 149, 'id': 29},
            {'text': 'to', 'start': 150, 'end': 152, 'id': 30},
            {'text': '1', 'start': 153, 'end': 154, 'id': 31},
            {'text': 'day', 'start': 155, 'end': 158, 'id': 32},
            {'text': ',', 'start': 158, 'end': 159, 'id': 33},
            {'text': 'tossing', 'start': 160, 'end': 167, 'id': 34},
            {'text': 'occasionally', 'start': 168, 'end': 180, 'id': 35},
            {'text': '.', 'start': 180, 'end': 181, 'id': 36}]


@pytest.fixture
def labels():
    return [{'start': 61, 'end': 66, 'token_start': 12, 'token_end': 12, 'label': 'FOOD'},
            {'start': 81, 'end': 85, 'token_start': 16, 'token_end': 16, 'label': 'FOOD'},
            {'start': 90, 'end': 96, 'token_start': 18, 'token_end': 18, 'label': 'FOOD'},
            {'start': 104, 'end': 109, 'token_start': 21, 'token_end': 21, 'label': 'FOOD'}]


@pytest.fixture
def tokens_with_labels():
    return [{'text': 'Combine', 'start': 0, 'end': 7, 'id': 0}, {'text': 'all', 'start': 8, 'end': 11, 'id': 1},
            {'text': 'ingredients', 'start': 12, 'end': 23, 'id': 2}, {'text': 'in', 'start': 24, 'end': 26, 'id': 3},
            {'text': 'large', 'start': 27, 'end': 32, 'id': 4}, {'text': 'bowl', 'start': 33, 'end': 37, 'id': 5},
            {'text': ';', 'start': 37, 'end': 38, 'id': 6}, {'text': 'toss', 'start': 39, 'end': 43, 'id': 7},
            {'text': 'to', 'start': 44, 'end': 46, 'id': 8}, {'text': 'blend', 'start': 47, 'end': 52, 'id': 9},
            {'text': '.', 'start': 52, 'end': 53, 'id': 10}, {'text': 'Season', 'start': 54, 'end': 60, 'id': 11},
            {'text': 'salsa', 'start': 61, 'end': 66, 'id': 12, 'label': 'FOOD'},
            {'text': 'to', 'start': 67, 'end': 69, 'id': 13}, {'text': 'taste', 'start': 70, 'end': 75, 'id': 14},
            {'text': 'with', 'start': 76, 'end': 80, 'id': 15},
            {'text': 'salt', 'start': 81, 'end': 85, 'id': 16, 'label': 'FOOD'},
            {'text': 'and', 'start': 86, 'end': 89, 'id': 17},
            {'text': 'pepper', 'start': 90, 'end': 96, 'id': 18, 'label': 'FOOD'},
            {'text': '.', 'start': 96, 'end': 97, 'id': 19}, {'text': 'Cover', 'start': 98, 'end': 103, 'id': 20},
            {'text': 'salsa', 'start': 104, 'end': 109, 'id': 21, 'label': 'FOOD'},
            {'text': 'and', 'start': 110, 'end': 113, 'id': 22},
            {'text': 'refrigerate', 'start': 114, 'end': 125, 'id': 23},
            {'text': 'at', 'start': 126, 'end': 128, 'id': 24}, {'text': 'least', 'start': 129, 'end': 134, 'id': 25},
            {'text': '2', 'start': 135, 'end': 136, 'id': 26}, {'text': 'hours', 'start': 137, 'end': 142, 'id': 27},
            {'text': 'and', 'start': 143, 'end': 146, 'id': 28}, {'text': 'up', 'start': 147, 'end': 149, 'id': 29},
            {'text': 'to', 'start': 150, 'end': 152, 'id': 30}, {'text': '1', 'start': 153, 'end': 154, 'id': 31},
            {'text': 'day', 'start': 155, 'end': 158, 'id': 32}, {'text': ',', 'start': 158, 'end': 159, 'id': 33},
            {'text': 'tossing', 'start': 160, 'end': 167, 'id': 34},
            {'text': 'occasionally', 'start': 168, 'end': 180, 'id': 35},
            {'text': '.', 'start': 180, 'end': 181, 'id': 36}]


def test_get_token_and_label_spans_from_jsonl(tokens, labels):
    sample_file = "../resources/food_sample.jsonl"

    expected_recid = '17006_1'

    actual_tokens, actual_labels, actual_records = get_token_and_label_spans_from_jsonl(sample_file)
    assert actual_tokens[0] == tokens
    assert actual_labels[0] == labels
    assert actual_records[0] == expected_recid
    assert len(actual_tokens) == 3
    assert len(actual_tokens) == 3
    assert len(actual_records) == 3


def test_assign_labels_to_tokens(tokens, labels, tokens_with_labels):
    assert assign_labels_to_tokens(tokens, labels) == tokens_with_labels


def test_tokens_overlap_1():
    token_1 = {'text': 'salsa', 'start': 61, 'end': 66, 'id': 12}
    token_2 = {'start': 61, 'end': 66, 'token_start': 12, 'token_end': 12, 'label': 'FOOD'}
    assert tokens_overlap(token_1, token_2) == True


def test_tokens_overlap_2():
    token_1 = {'start': 61, 'end': 66, 'token_start': 12, 'token_end': 12, 'label': 'FOOD'}
    token_2 = {'text': 'salsa', 'start': 64, 'end': 66, 'id': 12}
    assert tokens_overlap(token_1, token_2) == True


def test_tokens_overlap_3():
    token_1 = {'start': 61, 'end': 66, 'token_start': 12, 'token_end': 12, 'label': 'FOOD'}
    token_2 = {'text': 'occasionally', 'start': 168, 'end': 180, 'id': 35}
    assert tokens_overlap(token_1, token_2) == False


def test_assign_metadata_to_tokens(tokens):
    assert assign_metadata_to_tokens(tokens, "hello")[0] == {'text': 'Combine', 'start': 0, 'end': 7, 'id': 0,
                                                             "record_id": "hello"}


def test_add_bio_tags(tokens_with_labels):
    input = [{'text': 'Green', 'start': 0, 'end': 5, 'id': 0, 'label': "FOOD"},
             {'text': 'eggs', 'start': 6, 'end': 10, 'id': 1, 'label': "FOOD"},
             {'text': 'and', 'start': 11, 'end': 14, 'id': 2, },
             {'text': 'ham', 'start': 15, 'end': 18, 'id': 3, 'label': "FOOD"},
             {'text': '.', 'start': 18, 'end': 19, 'id': 4}]

    output = [{'text': 'Green', 'start': 0, 'end': 5, 'id': 0, 'label': "FOOD", 'bio-tag': "B-FOOD"},
              {'text': 'eggs', 'start': 6, 'end': 10, 'id': 1, 'label': "FOOD", 'bio-tag': "I-FOOD"},
              {'text': 'and', 'start': 11, 'end': 14, 'id': 2, 'bio-tag': "O"},
              {'text': 'ham', 'start': 15, 'end': 18, 'id': 3, 'label': "FOOD", 'bio-tag': "B-FOOD"},
              {'text': '.', 'start': 18, 'end': 19, 'id': 4, 'bio-tag': "O", 'bio-tag': "O"}]

    assert add_bio_tags(input) == output


def test_get_bio_tagged_spans_from_jsonl():
    sample_file = "../resources/food_sample.jsonl"
    spans = get_bio_tagged_spans_from_jsonl(sample_file)

    expected_spans = [{'text': 'Combine', 'start': 0, 'end': 7, 'id': 0, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'all', 'start': 8, 'end': 11, 'id': 1, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'ingredients', 'start': 12, 'end': 23, 'id': 2, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'in', 'start': 24, 'end': 26, 'id': 3, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'large', 'start': 27, 'end': 32, 'id': 4, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'bowl', 'start': 33, 'end': 37, 'id': 5, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': ';', 'start': 37, 'end': 38, 'id': 6, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'toss', 'start': 39, 'end': 43, 'id': 7, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'to', 'start': 44, 'end': 46, 'id': 8, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'blend', 'start': 47, 'end': 52, 'id': 9, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': '.', 'start': 52, 'end': 53, 'id': 10, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'Season', 'start': 54, 'end': 60, 'id': 11, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'salsa', 'start': 61, 'end': 66, 'id': 12, 'label': 'FOOD', 'record_id': '17006_1',
               'bio-tag': 'B-FOOD'},
              {'text': 'to', 'start': 67, 'end': 69, 'id': 13, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'taste', 'start': 70, 'end': 75, 'id': 14, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'with', 'start': 76, 'end': 80, 'id': 15, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'salt', 'start': 81, 'end': 85, 'id': 16, 'label': 'FOOD', 'record_id': '17006_1',
               'bio-tag': 'B-FOOD'},
              {'text': 'and', 'start': 86, 'end': 89, 'id': 17, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'pepper', 'start': 90, 'end': 96, 'id': 18, 'label': 'FOOD', 'record_id': '17006_1',
               'bio-tag': 'B-FOOD'},
              {'text': '.', 'start': 96, 'end': 97, 'id': 19, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'Cover', 'start': 98, 'end': 103, 'id': 20, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'salsa', 'start': 104, 'end': 109, 'id': 21, 'label': 'FOOD', 'record_id': '17006_1',
               'bio-tag': 'B-FOOD'},
              {'text': 'and', 'start': 110, 'end': 113, 'id': 22, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'refrigerate', 'start': 114, 'end': 125, 'id': 23, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'at', 'start': 126, 'end': 128, 'id': 24, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'least', 'start': 129, 'end': 134, 'id': 25, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': '2', 'start': 135, 'end': 136, 'id': 26, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'hours', 'start': 137, 'end': 142, 'id': 27, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'and', 'start': 143, 'end': 146, 'id': 28, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'up', 'start': 147, 'end': 149, 'id': 29, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'to', 'start': 150, 'end': 152, 'id': 30, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': '1', 'start': 153, 'end': 154, 'id': 31, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'day', 'start': 155, 'end': 158, 'id': 32, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': ',', 'start': 158, 'end': 159, 'id': 33, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'tossing', 'start': 160, 'end': 167, 'id': 34, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': 'occasionally', 'start': 168, 'end': 180, 'id': 35, 'record_id': '17006_1', 'bio-tag': 'O'},
              {'text': '.', 'start': 180, 'end': 181, 'id': 36, 'record_id': '17006_1', 'bio-tag': 'O'}]

    assert spans[0] == expected_spans