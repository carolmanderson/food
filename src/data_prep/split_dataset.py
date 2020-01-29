import random


def train_dev_test_split(sequence, train_frac, dev_frac):
    """Given a list-like object, divide into training, dev, and test sets.
    :param sequence: a list, tuple, etc.
    :param train_frac: float between 0 and 1. What proportion of the data to put in the training set
    :param dev_frac: float between 0 and 1. what proportion of the data to put in the dev set.
                    Cannot be greater than 1-train_frac.
    :returns three lists with non-overlapping elements (training, dev, and test sets)
    """
    if train_frac > 1 or train_frac < 0 or dev_frac > 1 or dev_frac < 0:
        raise ValueError("train and dev fractions must be between 0 and 1!")
    if train_frac + dev_frac > 1:
        raise ValueError("sum of train and dev fractions can't be greater than 1!")
    desired_train_num = int(len(sequence) * train_frac)
    desired_dev_num = int((len(sequence) * dev_frac))
    train = random.sample(sequence, desired_train_num)
    devtest = [item for item in sequence if item not in train]
    dev = random.sample(devtest, desired_dev_num)
    test = [item for item in devtest if item not in dev]
    assert len(train) + len(dev) + len(test) == len(sequence)
    return train, dev, test