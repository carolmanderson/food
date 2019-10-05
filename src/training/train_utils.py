import copy
import random


def generate_batch_indices(batch_size, sentence_length_dict_original):
    """

    :param batch_size:
    :param sentence_length_dict_original:
    :return:
    """

    # TODO: allow random seed to be passed as param


    def add_sentences_to_batch(batch_indices, batch_size, sentence_lengths, sentence_length_dict, current_index):
        """

        :param batch_indices:
        :param batch_size:
        :param sentence_lengths:
        :param sentence_length_dict:
        :param current_index:
        :return:
        """
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
        """

        :param sorted_list:
        :param current_length:
        :param index:
        :return:
        """
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


if __name__ == "__main__":
    sentence_length_dict = {9: [0, 12], 2: [1, 2, 21], 30: [3, 14, 25], 31: [4], 33: [5, 26], 25: [6], 40: [7], 28: [8],
                            37: [9], 27: [10], 1: [11], 26: [13], 35: [15, 22, 23], 39: [16], 34: [17], 15: [18],
                            16: [19], 10: [20], 32: [24]}

    batch_size = 4

    all_batches = []
    batch_generator = generate_batch_indices(batch_size, sentence_length_dict)

    for batch in batch_generator:
        print("NEW BATCH:", batch)
        all_batches.append(batch)
    print(all_batches)

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


    # TODO: test batch sizes