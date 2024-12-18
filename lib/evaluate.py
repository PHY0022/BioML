import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import os



def flip(y):
    print("Flipped.")
    return 1 - y



def ACC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)



def ROC_curve(y_test, y_proba, path='./'):
    tprs, fprs, thres = metrics.roc_curve(y_test, y_proba)
    auc = metrics.auc(fprs, tprs)
    plt.plot(fprs, tprs, label=f"ROC curve (AUC = {auc:.2f})")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(os.path.join(path, 'roc_curve.png'), dpi=1000)
    print(f"ROC curve saved at {os.path.join(path, 'roc_curve.png')}")
    plt.show()



def Result(y_test, y_pred):
    tprs, fprs, thresholds = metrics.roc_curve(y_test, 1 - y_pred)
    auc = metrics.auc(fprs, tprs)
    if auc < 0.5:
        tprs, fprs, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fprs, tprs)

    
    # Find the best threshold (closest to (0, 1))
    distances = np.sqrt((fprs - 0)**2 + (tprs - 1)**2)
    min_distance_index = np.argmin(distances)
    best_threshold = thresholds[min_distance_index]


    y_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    acc = ACC(y_test, y_pred)
    if acc < 0.5:
        y_pred = np.array([1 if pred < best_threshold else 0 for pred in y_pred], dtype=int)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, 1 - y_pred).ravel()
        acc = ACC(y_test, y_pred)


    if tp + fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    if tn + fp == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


    return acc, tn, fp, fn, tp, sensitivity, specificity, MCC, auc
    # result = [{'Accuracy': acc,'Sensitivity': sensitivity, 'Specificity': specificity, 'MCC': MCC, 'PPV': PPV, 'NPV': NPV}]
    # result_df = pd.DataFrame(result, dtype=float)
    # result_df.index = ['Result']
    # return result_df



def Evaluation(y_test, y_pred, path=''):
    tprs, fprs, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fprs, tprs)

    if auc < 0.5:
        y_pred = flip(y_pred)
        tprs, fprs, thresholds = metrics.roc_curve(y_test, y_pred)
        auc = metrics.auc(fprs, tprs)

    plt.plot(fprs, tprs, label=f"Model (AUC = {auc:.2f})")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if path != '':
        plt.savefig(os.path.join(path, 'roc_curve.png'), dpi=1000)
        print(f"ROC curve saved at {os.path.join(path, 'roc_curve.png')}")
    plt.show()



    # Find the best threshold (closest to (0, 1))
    distances = np.sqrt((fprs - 0)**2 + (tprs - 1)**2)
    min_distance_index = np.argmin(distances)
    best_threshold = thresholds[min_distance_index]
    print(f"Best threshold: {best_threshold}")

    # Classification report
    y_pred = [0 if pred > best_threshold else 1 for pred in y_pred]
    accuracy = sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_test)]) / len(y_test)
    print(f"Accuracy: {accuracy}")
    print("Classification report:")
    print(metrics.classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))



def ROC_curve_of_models(y_test, model_perform_dict, path=''):
    for model_name, y_proba in model_perform_dict.items():
        y_proba = np.array(y_proba)

        tprs, fprs, thres = metrics.roc_curve(y_test, y_proba)
        auc = metrics.auc(fprs, tprs)
        if auc < 0.5:
            y_proba = flip(y_proba)
            tprs, fprs, thresholds = metrics.roc_curve(y_test, y_proba)
            auc = metrics.auc(fprs, tprs)

        plt.plot(fprs, tprs, label=f"{model_name} (AUC = {auc:.2f})")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    if path != '':
        plt.savefig(os.path.join(path, 'roc_curve.png'), dpi=1000)
        print(f"ROC curve saved at {os.path.join(path, 'roc_curve.png')}")

    plt.show()



def ROC_curve_of_cross_validation(answers, model_perform_dict, path=''):
    for i, (model_name, y_proba) in enumerate(model_perform_dict.items()):
        y_proba = np.array(y_proba)

        tprs, fprs, thres = metrics.roc_curve(answers[i], y_proba)
        auc = metrics.auc(fprs, tprs)
        if auc < 0.5:
            y_proba = flip(y_proba)
            tprs, fprs, thresholds = metrics.roc_curve(answers[i], y_proba)
            auc = metrics.auc(fprs, tprs)

        plt.plot(fprs, tprs, label=f"{model_name} (AUC = {auc:.2f})")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    if path != '':
        plt.savefig(os.path.join(path, 'roc_curve.png'), dpi=1000)
        print(f"ROC curve saved at {os.path.join(path, 'roc_curve.png')}")

    plt.show()