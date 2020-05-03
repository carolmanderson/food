import pickle
import sys

import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy import displacy
import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from food_tools.training.dataset_utils import tokens_to_indices



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


if __name__ == "__main__":
    saved_model = "/Users/Carol/Google " \
                  "Drive/nlp_data/saved_models/20200421_food_ner.h5"

    saved_mappings = "/Users/Carol/Google " \
                     "Drive/nlp_data/saved_models/20200421_food_ner_mappings" \
                     ".pkl"

    HTML_WRAPPER = '<div style="overflow-x: auto; border: 1px solid #e6e9ef; ' \
                   'border-radius: 0.25rem; padding: 1rem; margin-bottom: ' \
                   '2.5rem">{}</div>'

    model, session = load_model(saved_model)
    mappings = load_mappings(saved_mappings)
    index_to_label = {v: k for k, v in mappings['label_to_index'].items()}
    token_to_index = mappings['token_to_index']
    sentencizer, tokenizer  = load_sentencizer_and_tokenizer()


    recipe_1 =  "Heat garlic and rosemary with oil. Drizzle oil over dip and serve with " \
                    "vegetables."
    recipe_2 = "Combine pineapple, banana, cream of coconut, rolled " \
                "oats, quick-cooking oats, baking powder, mint, " \
                "chia seeds, and poppy seeds in a blender; blend until " \
                "smooth. Pour into 2 mugs."

    recipe_dict = {"Veg" : recipe_1, "Cake": recipe_2}

    st.markdown("# Food finder")


    recipe = st.selectbox("Choose a sample recipe or enter your own",
                          list(recipe_dict.keys()), 0)
    text = st.text_area("Text to analyze", recipe_dict[recipe])

    sents = sentencizer(text)
    all_tokens = []
    for sent in sents.sents:
        tokens = tokenizer(sent.text)
        all_tokens.append([t.text for t in tokens])

    final_doc = None   # collects results from all sentences
    with session.graph.as_default():
        K.set_session(session)
        for tokens in all_tokens:
            token_indices = tokens_to_indices(tokens, token_to_index)
            preds = model.predict([tokens_to_indices(tokens, token_to_index)])
            preds = np.argmax(preds, axis=-1)
            labels = [index_to_label[ind[0]] for ind in preds]
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
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
