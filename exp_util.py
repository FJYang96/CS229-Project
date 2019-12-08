import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, auc, roc_auc_score

def classify_output(x):
    f = lambda x: 0 if x < 21 else (1 if x < 49 else 2)
    return np.array(list(map(f, x)))

def val_bin_accuracy(reg, xval, yval):
    assert yval.mean() > 1, "Works only for continuous labels"
    bin_pred = classify_output(reg.predict(xval))
    bin_label = classify_output(yval)
    return np.mean(bin_pred == bin_label)

def classify_output_two_classes(x):
    f = lambda x: 0 if x < 30 else 1
    return np.array(list(map(f, x)))

def val_bin_accuracy_two_classes(reg, xval, yval):
    assert yval.mean() > 1, "Works only for continuous labels"
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

def get_classification_metrics(TP, FP, TN, FN):
    sensitivity, specificity, precision_plus, precision_minus = \
        None, None, None, None
    try:
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision_plus = TP / (TP + FP)
        precision_minus = TN / (TN + FN)
    except:
        pass
    return sensitivity, specificity, precision_plus, precision_minus
    
def visualize_con_matrix(TP, FP, TN, FN):
    print('-'*41)
    print('|Pred\True\t|Pos\t\t|Neg\t|')
    print('-'*41)
    print('|Pos\t\t|',TP,'\t\t|',FP, '\t|')
    print('-'*41)
    print('|Neg\t\t|',FN,'\t\t|',TN, '\t|')
    print('-'*41)

def summarize_reg_performance(model, xval, yval):
    ypred = model.predict(xval)
    print('Validation score is:\t\t', model.score(xval, yval))
    print('Mean squared error is:\t\t', mean_squared_error(yval, ypred))
    binned_results = val_bin_accuracy_with_con_matrix(model, xval, yval)
    print('Classification accuracy is:\t', binned_results[0])
    class_metrics = get_classification_metrics(*binned_results[1:])
    print('Sensitivity:\t\t\t', class_metrics[0])
    print('Specificity:\t\t\t', class_metrics[1])
    print('Precision+:\t\t\t', class_metrics[2])
    print('Precision-:\t\t\t', class_metrics[3])
    visualize_con_matrix(*binned_results[1:])

def summarize_clf_performance(model, xval, yval):
    print('Mean accuracy is:\t\t', model.score(xval, yval))
    pred = model.predict(xval)
    TN, FP, FN, TP = confusion_matrix(yval,pred).ravel()
    class_metrics = get_classification_metrics(TP, FP, TN, FN)
    print('Sensitivity:\t\t\t', class_metrics[0])
    print('Specificity:\t\t\t', class_metrics[1])
    print('Precision+:\t\t\t', class_metrics[2])
    print('Precision-:\t\t\t', class_metrics[3])
    visualize_con_matrix(TP, FP, TN, FN)

def param_search_and_analyze(model, params, Xcv, ycv, xval, yval):
    gridcv = GridSearchCV(
        estimator=model, param_grid=params, cv=10)
    gridcv.fit(Xcv, ycv)
    best_model = gridcv.best_estimator_
    summarize_performance(best_model, xval, yval)
    return best_model

def summarize_performance(model, xval, yval):
    isregression = yval.mean() > 1
    if isregression:
        summarize_reg_performance(model, xval, yval)
    else:
        summarize_clf_performance(model, xval, yval)
