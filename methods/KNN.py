import warnings
from matplotlib import pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import zero_one_loss, classification_report, confusion_matrix, plot_confusion_matrix, roc_curve, \
    auc
from sklearn.model_selection import cross_val_score, GridSearchCV
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
def knnClassifier(x_train, x_test, y_train, y_test, y, infoPrint, genplot):
    # KNN
    if infoPrint:
        print("\n-- KNN")

    # definizione e training del classificatore KNN
    knc = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=5, p=2, weights='distance').fit(x_train, y_train)
    y_pred_knc = knc.predict(x_test)
    ncv_score_knc = knc.score(x_test, y_test)
    score_train_knc = knc.score(x_train, y_train)
    score_knc = cross_val_score(knc, x_train, y_train, cv=5)
    knc_fpr, knc_tpr, thr = roc_curve(y_test, y_pred_knc)
    accuracy = auc(knc_fpr, knc_tpr)
    if infoPrint:
        print("Train score: ", score_train_knc, "\nTest score: ", ncv_score_knc)
        print("Cross Validated Score: ", score_knc.mean())
        print("Standard Deviation: ", score_knc.std())
        print("Variance: ", np.var(score_knc))
        print("0-1 Loss: ", zero_one_loss(y_test, y_pred_knc))
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred_knc))


    if genplot:
        plot_confusion_matrix(knc,
                              x_test,
                              y_test,
                              values_format='d',
                              display_labels=['Nondemented', 'Demented'])
        plt.title("K-Nearest Neighbors")
        plt.savefig("images/after_opt/K-NearestNeighbors.png")
    return knc, knc_fpr, knc_tpr

def kncWithGridView(x_train, y_train):
    print("-- Execute KNC with Grid View")
    tic = time.perf_counter()
    param_grid = {'n_neighbors': [5,10,15,30,60,90,120], 'weights': ['uniform','distance'], 'algorithm':['kd_tree','ball_tree','brute'],'p':[1,2]}
    search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, refit=True, error_score=0, n_jobs=-1, return_train_score=True)
    search.fit(x_train, y_train)
    toc = time.perf_counter()
    return search, toc-tic

