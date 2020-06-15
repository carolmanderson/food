import argparse
import ast
import pickle

import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy import displacy
import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
import yaml

from food_tools.training.dataset_utils import \
    tokens_to_indices, correct_BIO_encodings



def output_to_displacy(tokens, labels):
    text = ""
    start = 0
    ents = []
    curr_label = ""
    new_ent = {}
    for token, label in zip(tokens, labels):
        text += token + " "
        end = start + len(token)
        if label.startswith("B-"):
            if new_ent:
                ents.append(new_ent)
            curr_label = label[2:]
            new_ent = {"start": start, "end": end,
                       "label": curr_label}
        elif label.startswith("I-"):
            assert label[2:] == curr_label
            new_ent['end'] = end
        elif label == "O":
            if new_ent:
                ents.append(new_ent)
                new_ent = {}
        else:
            raise Exception("Found non-BIO label {}!".format(label))
        start += len(token) + 1
    if new_ent:
        ents.append(new_ent)
    doc = {"text": text,
           "ents": ents,
           "title": None}
    return doc


@st.cache(allow_output_mutation=True)
def load_model(model_path):
    """
    Load NER model
    https://github.com/tensorflow/tensorflow/issues/14356
    https://github.com/tensorflow/tensorflow/issues/28287
    """
    session = tf.Session(graph=tf.Graph())
    with session.graph.as_default():
        K.set_session(session)
        loaded_model = tf.keras.models.load_model(model_path)
        loaded_model.summary()
    return loaded_model, session


@st.cache(allow_output_mutation=True)
def load_mappings(filepath):
    """
    Token mappings for NER model
    """
    return pickle.load(open(filepath, "rb"))


@st.cache(hash_funcs={Tokenizer: id, English:id})
def load_sentencizer_and_tokenizer():
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    return nlp, tokenizer


def form_matrix(tokens):
    tokens = np.expand_dims(tokens, axis=0)
    return np.array(tokens)


def get_tokens(input_cell):
    """For use in custom count vectorizer"""
    return ast.literal_eval(input_cell)


def extract_food_terms(tokens, labels):
    # get a list of food terms from NER prediction for classifier
    text = ""
    start = 0
    ents = []
    curr_label = ""
    new_ent = {}
    for token, label in zip(tokens, labels):
        text += token + " "
        end = start + len(token)
        if label.startswith("B-"):
            if new_ent:
                new_ent['text'] = text[new_ent['start']:new_ent['end']]
                ents.append(new_ent)
            curr_label = label[2:]
            new_ent = {"start": start, "end": end,
                       "label": curr_label}
        elif label.startswith("I-"):
            assert label[2:] == curr_label
            new_ent['end'] = end
        elif label == "O":
            if new_ent:
                new_ent['text'] = text[new_ent['start']:new_ent['end']]
                ents.append(new_ent)
                new_ent = {}
        else:
            raise Exception("Found non-BIO label {}!".format(label))
        start += len(token) + 1
    if new_ent:
        new_ent['text'] = text[new_ent['start']:new_ent['end']]
        ents.append(new_ent)
    return ents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file")

    args = parser.parse_args()

    with open(args.config) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    HTML_WRAPPER = '<div style="overflow-x: auto; border: 1px solid #e6e9ef; ' \
                   'border-radius: 0.25rem; padding: 1rem; margin-bottom: ' \
                   '2.5rem">{}</div>'

    #### NER MODEL ###########
    model, session = load_model(params['ner_model'])
    mappings = load_mappings(params['ner_mappings'])

    index_to_label = {v: k for k, v in mappings['label_to_index'].items()}
    token_to_index = mappings['token_to_index']
    sentencizer, tokenizer  = load_sentencizer_and_tokenizer()

    ####  CLASSIFIERS ##############
    vegan_model = pickle.load(open(params['vegan_model'], "rb"))
    kosher_model = pickle.load(open(params['kosher_model'], "rb"))
    gf_model = pickle.load(open(params['gf_model'], "rb"))
    vectorizer = pickle.load(open(params['vectorizer'], "rb"))

    ####################################

    recipe_1 = "Scrub the potatoes and boil them in their jackets. Finely " \
               "chop the scallions. Cover the scallions with cold milk and " \
               "bring slowly to a boil. Simmer for about 3 to 4 minutes, " \
               "then turn off the heat and leave to infuse. Peel and mash " \
               "the freshly boiled potatoes and, while hot, mix with the " \
               "boiling milk and scallions. Beat in some of the butter. " \
               "Season to taste with salt and freshly ground pepper. Serve " \
               "in one large or four individual bowls with a knob of butter " \
               "melting in the center."
    recipe_2 = "Heat oil in heavy large skillet over high heat. Add ginger " \
               "and garlic; sauté 20 seconds. Add bell pepper and mushrooms; " \
               "sauté until pepper is crisp-tender, about 3 minutes. Add " \
               "green onions and bok choy and sauté until just wilted, " \
               "about 2 minutes. Season with salt and pepper."
    recipe_3 = "Finely mash tofu with a fork in a bowl, then let drain in a " \
               "sieve set over another bowl, about 15 minutes (discard " \
               "liquid). While tofu drains, whisk together mayonnaise, " \
               "lemon juice, turmeric, and mustard in bowl, then stir in " \
               "tofu, celery, chives, salt, and pepper."
    recipe_4 = "In a blender blend at high speed the bananas, the pineapple " \
               "juice, the orange juice, the yogurt, and the honey until the " \
               "mixture is smooth and frothy and divide the mixture among " \
               "chilled glasses."
    recipe_5 = "Cook bacon in heavy large skillet over medium heat until " \
               "brown and crisp. Using slotted spoon, transfer bacon to " \
               "paper towel and drain. Pour off all but 2 tablespoons " \
               "drippings from skillet. Add green beans and bell pepper to " \
               "skillet. Toss vegetables over medium high heat until coated " \
               "with drippings, about 1 minute. Add broth. Cover and cook " \
               "until vegetables are crisp tender, about 5 minutes. Season " \
               "to taste with salt and pepper. Transfer to serving bowl. " \
               "Sprinkle with bacon and serve."
    recipe_6 = 'Bring a large saucepan of salted water to a boil for ' \
               'pasta.In a small saucepan cook garlic in oil over moderately ' \
               'low heat, stirring, until softened and transfer mixture to a ' \
               'large bowl. Add pine nuts, crushing them lightly with the ' \
               'back of a fork, lemon juice, zest, parsley, and salt and ' \
               'pepper to taste.Cook pasta in boiling water until al dente ' \
               'and, reserving 2 tablespoons of the cooking water, ' \
               'drain well. Add pasta with the reserved cooking water to ' \
               'bowl and toss with lemon mixture until it is absorbed. Serve ' \
               'pasta warm or at room temperature.'

    recipe_dict = {"Mashed Potatoes": recipe_1, "Thai Vegetables": recipe_2,
                   "Creamy Tofu Salad": recipe_3, "Banana Smoothie":
                       recipe_4, "Green Beans with Bacon and Red Bell "
                                 "Pepper": recipe_5, "Angel Hair Pasta with "
                                                     "Lemon and Pine Nuts":
                       recipe_6}

    st.markdown("# Recipe Analyzer")


    recipe = st.selectbox("Choose a sample recipe or enter your own",
                          list(recipe_dict.keys()), 0)
    text = st.text_area("Text to analyze", recipe_dict[recipe])

    sents = sentencizer(text)
    all_tokens = []
    for sent in sents.sents:
        tokens = tokenizer(sent.text)
        all_tokens.append([t.text for t in tokens])

    final_doc = None   # collects entity spans from all sentences
    all_terms = []  # collects food terms from all sentences
    with session.graph.as_default():
        K.set_session(session)
        for tokens in all_tokens:
            token_indices = tokens_to_indices(tokens, token_to_index)
            preds = model.predict([tokens_to_indices(tokens, token_to_index)])
            preds = np.argmax(preds, axis=-1)
            labels = [index_to_label[ind[0]] for ind in preds]
            labels = correct_BIO_encodings(labels)
            terms = [term['text'] for term in
                     extract_food_terms(tokens, labels)]
            all_terms.extend(terms)
            doc = output_to_displacy(tokens, labels)
            if not final_doc:  # first sentence
                final_doc = doc
                continue
            shift = len(final_doc['text'])
            for ent in doc['ents']:
                ent['start'] += shift
                ent['end'] += shift
                final_doc['ents'].append(ent)
            final_doc['text'] += doc['text']

    colors = {"FOOD": "#87CEEB"}
    options = {"ents": ["FOOD"], "colors": colors}
    html = displacy.render(final_doc, style="ent", options={"colors":colors},
                           manual=True)
    html = html.replace("\n", " ")
    st.markdown("### NER results: ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    features = vectorizer.transform([str(all_terms)])
    vegan_prob = int(round(100*vegan_model.predict_proba(features)[0][1]))
    kosher_prob = int(round(100*kosher_model.predict_proba(features)[0][1]))
    gf_prob = int(round(100*gf_model.predict_proba(features)[0][1]))

    st.markdown("### Classification results:")

    if vegan_prob < 50:
        st.error("Likely not vegan ({}% chance)".format(
            vegan_prob))
    else:
        st.success("Likely vegan ({}% chance)".format(vegan_prob))

    if kosher_prob < 50:
        st.error("Likely not kosher ({}% chance)".format(
            kosher_prob))
    else:
        st.success("Likely kosher ({}% chance)".format(kosher_prob))

    if gf_prob < 50:
        st.error("Likely not gluten-free ({}% chance)".format(
            gf_prob))
    else:
        st.success("Likely gluten-free ({}% chance)".format(gf_prob))
