import pytest

from src.data_prep.split_dataset import train_dev_test_split



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