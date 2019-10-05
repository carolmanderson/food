import pytest
from src.training.dataset_utils import compile_vocabulary, train_dev_test_split

@pytest.fixture
def dataset():
    return [
        [['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'],
         ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O']], [['Peter', 'B-PER'], ['Blackburn', 'I-PER']],
        [['BRUSSELS', 'B-LOC'], ['1996-08-22', 'O']],
        [['The', 'O'], ['European', 'B-ORG'], ['Commission', 'I-ORG'], ['said', 'O'], ['on', 'O'],
         ['Thursday', 'O'], ['it', 'O'], ['disagreed', 'O'], ['with', 'O'], ['German', 'B-MISC'], ['advice', 'O'],
         ['to', 'O'], ['consumers', 'O'], ['to', 'O'], ['shun', 'O'], ['British', 'B-MISC'], ['lamb', 'O'],
         ['until', 'O'], ['scientists', 'O'], ['determine', 'O'], ['whether', 'O'], ['mad', 'O'], ['cow', 'O'],
         ['disease', 'O'], ['can', 'O'], ['be', 'O'], ['transmitted', 'O'], ['to', 'O'], ['sheep', 'O'],
         ['.', 'O']]]

def test_lowercase(dataset):
    vocab = {'eu': 1, 'rejects': 1, 'german': 2, 'call': 1, 'to': 4, 'boycott': 1, 'british': 2, 'lamb': 2, '.': 2,
             'peter': 1, 'blackburn': 1, 'brussels': 1, '1996-08-22': 1, 'the': 1, 'european': 1, 'commission': 1,
             'said': 1, 'on': 1, 'thursday': 1, 'it': 1, 'disagreed': 1, 'with': 1, 'advice': 1, 'consumers': 1,
             'shun': 1, 'until': 1, 'scientists': 1, 'determine': 1, 'whether': 1, 'mad': 1, 'cow': 1, 'disease': 1,
             'can': 1, 'be': 1, 'transmitted': 1, 'sheep': 1}

    compiled_vocab = compile_vocabulary(dataset, lowercase=True)
    print(compiled_vocab)
    assert vocab == compiled_vocab


def test_originalcase(dataset):
    vocab = {'EU': 1, 'rejects': 1, 'German': 2, 'call': 1, 'to': 4, 'boycott': 1, 'British': 2, 'lamb': 2, '.': 2,
             'Peter': 1, 'Blackburn': 1, 'BRUSSELS': 1, '1996-08-22': 1, 'The': 1, 'European': 1, 'Commission': 1,
             'said': 1, 'on': 1, 'Thursday': 1, 'it': 1, 'disagreed': 1, 'with': 1, 'advice': 1, 'consumers': 1,
             'shun': 1, 'until': 1, 'scientists': 1, 'determine': 1, 'whether': 1, 'mad': 1, 'cow': 1, 'disease': 1,
             'can': 1, 'be': 1, 'transmitted': 1, 'sheep': 1}

    compiled_vocab = compile_vocabulary(dataset)
    print(compiled_vocab)
    assert vocab == compiled_vocab



@pytest.fixture
def simple_list():
    return list(range(10))

def test_train_dev_test_split_1(simple_list):
    train, dev, test = train_dev_test_split(simple_list, 0.6, 0.2)
    assert len(train) == 6
    assert len(dev) == 2
    assert len(test) == 2
    assert len(set(train).intersection(set(dev))) == 0
    assert len(set(train).intersection(set(test))) == 0
    assert len(set(test).intersection(set(dev))) == 0


def test_train_dev_test_split_2(simple_list):
    train, dev, test = train_dev_test_split(simple_list, 1, 0)
    assert len(train) == 10
    assert len(dev) == 0
    assert len(test) == 0


def test_train_dev_test_split_3(simple_list):
    train, dev, test = train_dev_test_split(simple_list, 0.6, 0.1)
    assert len(train) == 6
    assert len(dev) == 1
    assert len(test) == 3
    assert len(set(train).intersection(set(dev))) == 0
    assert len(set(train).intersection(set(test))) == 0
    assert len(set(test).intersection(set(dev))) == 0


def test_train_dev_test_split_4(simple_list):
    with pytest.raises(ValueError):
        train_dev_test_split(simple_list, 6, 1)


def test_train_dev_test_split_5(simple_list):
    with pytest.raises(ValueError):
        train_dev_test_split(simple_list, 0.6, 0.5)