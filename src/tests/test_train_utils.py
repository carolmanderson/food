import pytest
from src.training.train_utils import make_sentence_length_dict, generate_batch_indices

@pytest.fixture
def sentence_lengths():
    return [9, 2, 2, 30, 31, 33, 25, 40, 28, 37, 27, 1, 9, 26, 30, 35, 39, 34, 15, 16, 10, 2, 35, 35, 32, 30, 33]


def test_make_sentence_length_dict(sentence_lengths):
    expected_dict = {9: [0, 12], 2: [1, 2, 21], 30: [3, 14, 25], 31: [4], 33: [5, 26], 25: [6], 40: [7], 28: [8],
                            37: [9], 27: [10], 1: [11], 26: [13], 35: [15, 22, 23], 39: [16], 34: [17], 15: [18],
                            16: [19], 10: [20], 32: [24]}
    assert make_sentence_length_dict(sentence_lengths) ==  expected_dict


def test_generate_batch_indices_1(sentence_lengths):
    """Test that all indices in the data set get included in a batch"""
    batch_generator = generate_batch_indices(4, sentence_lengths, random_seed=2)
    all_batches = []
    all_indices_in_batches = []
    for batch in batch_generator:
        all_batches.append(batch)

    for batch in all_batches:
        for item in batch:
            all_indices_in_batches.append(item)
    all_indices_in_batches.sort()
    assert all_indices_in_batches == list(range(len(sentence_lengths)))

def test_generate_batch_indices_2(sentence_lengths):
    """Test that the first batch contains the expected items"""
    batch_generator = generate_batch_indices(4, sentence_lengths, random_seed=2)
    sample_batch = next(batch_generator)
    assert sample_batch == [1, 2, 21, 11]