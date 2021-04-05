import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef


def classification_metrics(predictions, labels):
    y_true=labels.cpu().numpy()
    y_pred=predictions.cpu().numpy()
    
    report=classification_report(y_true,y_pred)
    return report

def matthews_CC_score(labels, predictions):
    y_true=labels.cpu().numpy()
    y_pred=predictions.cpu().numpy()

    MCC=matthews_corrcoef(y_true, y_pred)
    return MCC


def confusion_matrix(predictions, labels):
    pred=predictions
    actual=labels
    actual_pred = torch.stack((actual, pred), dim=1)
    confusion_matrix = torch.zeros(10,10, dtype=torch.int32)
    for result in actual_pred.long():
        true, predicted = result.tolist()
        confusion_matrix[true,predicted] += 1
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, title):
    CM=confusion_matrix.numpy()
    fig=plt.figure()
    plt.imshow(CM, cmap=plt.cm.Blues, vmax=30)
    plt.title(title)
    classes=["action","anime","children","comedy","crime","documentary","drama","horror","international","romance"]
    tick_marks = np.arange(len(classes))
    for i in range(CM.shape[0]):
            for j in range(CM.shape[1]):
                plt.text(j, i, format(CM[i, j]), fontsize=8,
                         ha="center", va="center",
                         color="white" if CM[i,j]>50 else "black")
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig