import copy
import json


def get_token_and_label_spans_from_jsonl(db_file):
    """
    Given a Prodigy jsonl file containing tokens and labels,
    return three parallel lists: token spans, label spans, and record ids.
    :param db_file: the exported Prodigy database file
    :returns token_spans: a list of lists in which each list is one document.
                            Within each document, each token is a dict:
                            {'text': 'Season', 'start': 54, 'end': 60, 'id': 11}
            label_spans: a list of lists in which each list is one document.
                            Within each document, each token is a dict:
                            {'start': 61, 'end': 66, 'token_start': 12, 'token_end': 12, 'label': 'FOOD'}
            record_ids: a list of strings, which are the record IDs for each document
    """
    token_spans = []
    label_spans = []
    record_ids = []
    with open(db_file, 'r') as infile:
        for line in infile:
            record = json.loads(line)
            token_spans.append(record['tokens'])
            label_spans.append(record['spans'])
            num = record['meta']['row']
            part = record['meta']['subpart']
            record_ids.append(str(num) + "_" + str(part))
    return token_spans, label_spans, record_ids


def tokens_overlap(token_1, token_2):
    """In Prodigy's format, labels aren't recorded for specific tokens. Instead
    just their indices are saved, and these have to be matched up with token indices.
    This function helps to determine whether a given token lines up with a given label.
    :param token_1: a dict representing a token span or label span. Must have 'start' and 'end' as keys.
                    Examples:
                    {'text': 'Season', 'start': 54, 'end': 60, 'id': 11}
                    {'start': 61, 'end': 66, 'token_start': 12, 'token_end': 12, 'label': 'FOOD'}
    :returns boolean: True if tokens overlap, otherwise False

    """
    token_1_range = {i for i in range(token_1['start'], token_1['end'])}
    token_2_range = {i for i in range(token_2['start'], token_2['end'])}
    if token_1_range.intersection(token_2_range):
        return True
    return False


def assign_labels_to_tokens(token_spans, label_spans):
    """
    In Prodigy's format, labels and token spans are saved in separate part of each record.
    This function assigns a label to a token if there is any overlap between the token
    and label. TODO: this could be refactored to use the token indices in the labels
     ('token_start': 12, 'token_end': 12 indicates the label applies to the 13th token),
    instead of iterating through all labels for every token. This would be much more efficient,
    but also less flexible, since it would force use of the Prodigy tokenization.
    :param token_spans: a list of lists in which each list is one document.
                            Within each document, each token is a dict:
                            {'text': 'Season', 'start': 54, 'end': 60, 'id': 11}
    :param label_spans: a list of lists in which each list is one document.
                            Within each document, each token is a dict:
                            {'start': 61, 'end': 66, 'token_start': 12, 'token_end': 12, 'label': 'FOOD'}
    :returns a copy of token_spans with additional key "label" (for tokens that are in entities)
    """
    token_spans = copy.deepcopy(token_spans)
    for tok in token_spans:
        for lab in label_spans:
            if tokens_overlap(tok, lab):
                tok['label'] = lab['label']
    return token_spans


def assign_metadata_to_tokens(token_spans, record_id):
    """
    For purposes of converting Prodigy to CoNLL format, this associates
    the source document name with each token span obtained from assign_labels_to_tokens.
    :param token_spans: a list of lists in which each list is one document.
                        Within each document, each token is a dict:
                        {'text': 'Season', 'start': 54, 'end': 60, 'id': 11}
    :param record_id: a string to use as the record id
    :returns a copy of token_spans, with additional key 'record_id'
    """
    token_spans = copy.deepcopy(token_spans)
    for token in token_spans:
        token['record_id'] = record_id
    return token_spans


def add_bio_tags(token_spans):
    """
    Given a set of token spans from assign_labels_to_tokens, convert
    labels to BIO format.
    :param token_spans: a list of lists in which each list is one document.
                    Within each document, each token is a dict:
                    {'text': 'salsa', 'start': 54, 'end': 60, 'id': 11, 'label': "FOOD"}
                    Note that the labels must have already been added to the token spans.
    :returns a copy of the input token_spans, with additional key 'bio-tag'
    """
    token_spans = copy.deepcopy(token_spans)
    prev_label = 'O'
    for tok in token_spans:
        label = tok.get('label', 'O')
        if label == 'O':
            tok['bio-tag'] = 'O'
            prev_label = 'O'
            continue
        elif label == prev_label:
            tok['bio-tag'] = "I-" + label
            prev_label = label
        else:
            tok['bio-tag'] = "B-" + label
            prev_label = label
    return token_spans


def get_bio_tagged_spans_from_jsonl(db_file):
    """
    Given a Prodigy jsonl file, return a nested list of token spans where
    each sublist is a document/sentence, and within each document, each token is a dict such
    as {'text': 'salsa', 'start': 54, 'end': 60, 'record_id': myfilename, 'label': "FOOD", 'bio-tag': "B-FOOD"}
    The output is suitable for writing to a CoNLL-format file.
    """
    result = []
    token_spans, label_spans, record_ids = get_token_and_label_spans_from_jsonl(db_file)
    for tok, lab, rec_id in zip(token_spans, label_spans, record_ids):
        tagged_spans = assign_labels_to_tokens(tok, lab)
        tagged_spans = assign_metadata_to_tokens(tagged_spans, rec_id)
        bio_tagged_spans = add_bio_tags(tagged_spans)
        result.append(bio_tagged_spans)
    return result


def write_conll(bio_tagged_spans, outfile):
    """
    Given token spans as output by get_bio_tagged_spans_from_jsonl,
    write them to a CoNLL-formatted file (space delimited, one token per line).
    """
    with open(outfile, 'w') as out:
        for sent in bio_tagged_spans:
            for tok in sent:
                out.write(f"{tok['text']} {tok['start']} {int(tok['end']) + len(tok['text'])} {tok['record_id']} {tok['bio-tag']}\n")
            out.write("\n")


if __name__ == "__main__":
    db_file = "/Users/Carol/Documents/epicurious-recipes-with-rating-and-nutrition/labeled_data/food_ner_gold_2.jsonl"
    outfile = "/Users/Carol/Documents/Repos/food/src/resources/food_sample.jsonl"

    spans = get_bio_tagged_spans_from_jsonl(outfile)
    print(len(spans))
    print(spans[0])
