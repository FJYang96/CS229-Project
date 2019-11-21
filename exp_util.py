import numpy as np

def classify_output(x):
    f = lambda x: 0 if x < 21 else (1 if x < 49 else 2)
    return np.array(list(map(f, x)))

def val_bin_accuracy(reg, xval, yval):
    bin_pred = classify_output(reg.predict(xval))
    bin_label = classify_output(yval)
    return np.mean(bin_pred == bin_label)

def classify_output_two_classes(x):
    f = lambda x: 0 if x < 30 else 1
    return np.array(list(map(f, x)))

def val_bin_accuracy_two_classes(reg, xval, yval):
    bin_pred = classify_output_two_classes(reg.predict(xval))
    bin_label = classify_output_two_classes(yval)
    return np.mean(bin_pred == bin_label)


def val_bin_accuracy_with_con_matrix(reg, xval, yval):
    bin_pred = classify_output_two_classes(reg.predict(xval))
    bin_label = classify_output_two_classes(yval)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
        
    for i in range(len(yval)):
        if ((bin_pred[i] == 1) & (bin_label[i] == 1)):
            TP = TP + 1
        if ((bin_pred[i] == 1) & (bin_label[i] == 0)):
            FP = FP + 1
        if ((bin_pred[i] == 0) & (bin_label[i] == 0)):
            TN = TN + 1
        if ((bin_pred[i] == 0) & (bin_label[i] == 1)):
            FN = FN + 1

    return (np.mean(bin_pred == bin_label), TP, FP, TN, FN)

def get_classification_metrics(reg, xval, yval):
    accuracy, TP, FP, TN, FN = val_bin_accuracy_with_con_matrix(reg, xval, yval)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision_plus = TP / (TP + FP)
    precision_minus = TN / (TN + FN)
    
    return accuracy, sensitivity, specificity, precision_plus, precision_minus
    
