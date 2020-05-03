import numpy as np
from sklearn.metrics import classification_report

from src.training.train_utils import form_ner_pred_matrix

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