import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss, classification_report, confusion_matrix, roc_curve, auc, \
    plot_confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import time

def rfClassifier(x_train, x_test, y_train, y_test, y, infoPrint, genplot):
    if infoPrint:
        print("\n-- RANDOM FOREST CLASSIFIER")
    rfc = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=9, max_features='sqrt', n_estimators=30)
    rfc.fit(x_train, y_train)
    y_pred_rfc = rfc.predict(x_test)
    ncv_score_rfc = rfc.score(x_test, y_test)
    score_train_rfc = rfc.score(x_train, y_train)
    score_rfcl = cross_val_score(rfc, x_train, y_train, cv=20)
    rfc_fpr, rfc_tpr, thr = roc_curve(y_test, y_pred_rfc)
    accuracy = auc(rfc_fpr, rfc_tpr)
    if infoPrint:
        print("Train score: ", score_train_rfc, "\nTest score: ", ncv_score_rfc)
        print("Cross Validated Score: ", score_rfcl.mean())
        print("Standard Deviation: ", score_rfcl.std())
        print("Variance: ", np.var(score_rfcl))
        print("0-1 Loss: ", zero_one_loss(y_test, y_pred_rfc))
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred_rfc))
    if genplot:
        plot_confusion_matrix(rfc,
                              x_test,
                              y_test,
                              values_format='d',
                              display_labels=['Nondemented', 'Demented'])
        plt.title("Random Forest")
        plt.savefig("images/after_opt/RandomForest.png")

    return rfc, rfc_fpr, rfc_tpr


def rfcWithGridView(x_train, y_train):
    print("-- Execute RFC with Grid View")
    tic = time.perf_counter()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]

    # Number of features to consider at every split
    max_features = ['sqrt', 'log2']

    # Maximum number of levels in tree
    max_depth = range(1, 10)

    # measure the quality of a split
    criterion = ['gini']

    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the param grid
    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'criterion': criterion,
                  'bootstrap': bootstrap}

    optimal_params = GridSearchCV(RandomForestClassifier(),
                                  param_grid,
                                  cv=5,  # we are taking 5-fold as in k-fold cross validation
                                  scoring='accuracy',  # try the other scoring if have time
                                  verbose=0,
                                  n_jobs=-1)

    optimal_params.fit(x_train, y_train)
    toc = time.perf_counter()
    return optimal_params, toc-tic
