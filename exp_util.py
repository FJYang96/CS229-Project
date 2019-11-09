import numpy as np

def classify_output(x):
    f = lambda x: 0 if x < 21 else (1 if x < 49 else 2)
    return np.array(list(map(f, x)))

def val_bin_accuracy(reg, xval, yval):
    bin_pred = classify_output(reg.predict(xval))
    bin_label = classify_output(yval)
    return np.mean(bin_pred == bin_label)
