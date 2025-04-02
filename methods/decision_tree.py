import warnings
import time
import numpy as np
from matplotlib import pyplot as plt
from plotly.offline import plot
from sklearn.metrics import zero_one_loss, classification_report, confusion_matrix, plot_confusion_matrix, roc_curve, \
    auc
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")


def dtClassifier(x_train, x_test, y_train, y_test, y, infoPrint, genplot):
    # DECISION TREE CLASSIFIER
    if infoPrint:
        print("\n-- DECISION TREE CLASSIFIER")

    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=7).fit(x_train, y_train)
    y_pred_dct = dtc.predict(x_test)
    ncv_score_dct = dtc.score(x_test, y_test)
    score_train_dct = dtc.score(x_train, y_train)
    score_dct = cross_val_score(dtc, x_train, y_train, cv=10)
    dtc_fpr, dtc_tpr, thr = roc_curve(y_test, y_pred_dct)
    accuracy = auc(dtc_fpr, dtc_tpr)
    if infoPrint:
        print("Train score: ", score_train_dct, "\nTest score: ", ncv_score_dct)
        print("Cross Validated Score: ", score_dct.mean(), "\nStandard Deviation: ",
              score_dct.std(), "\nVariance  for Decision Tree: ", np.var(score_dct))
        print("0-1 Loss: ", zero_one_loss(y_test, y_pred_dct))
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred_dct))

    if genplot:
        plot_confusion_matrix(dtc,
                              x_test,
                              y_test,
                              values_format='d',
                              display_labels=['Nondemented', 'Demented'])
        plt.figure(figsize=(5, 5), dpi=100)
        plt.title("Decision Tree")
        plt.savefig("images/after_opt/DecisionTree.png")

    return dtc, dtc_fpr, dtc_tpr


def dtcWithGridView(x_train, y_train):
    print("-- Execute DCT with Grid View")
    tic = time.perf_counter()
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': range(1, 100)}
    optimal_params  = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, refit=True, error_score=0,
                          n_jobs =-1, return_train_score=True)
    optimal_params .fit(x_train, y_train)
    toc = time.perf_counter()
    return optimal_params, toc-tic
