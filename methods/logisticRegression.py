import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix, zero_one_loss, classification_report
import time
# LOGISTIC REGRESSION classifier
def log_reg(x_train, x_test, y_train, y_test, y, infoPrint, genplot):
    if infoPrint:
        print("\n-- LOGISTIC REGRESSION")
    lr_model = LogisticRegression(C=5, penalty='l2').fit(x_train, y_train)
    y_pred_rfc = lr_model.predict(x_test)
    ncv_score_rfc = lr_model.score(x_test, y_test)
    score_train_rfc = lr_model.score(x_train, y_train)
    score_rfcl = cross_val_score(lr_model, x_train, y_train, cv=20)
    lgr_fpr, lgr_tpr, thr = roc_curve(y_test, y_pred_rfc)
    accuracy = auc(lgr_fpr, lgr_tpr)
    if infoPrint:
        print("Train score: ", score_train_rfc, "\nTest score: ", ncv_score_rfc)
        print("Cross Validation Score: ", score_rfcl.mean())
        print("Standard Deviation: ", score_rfcl.std())
        print("Variance: ", np.var(score_rfcl))
        print("0-1 Loss: ", zero_one_loss(y_test, y_pred_rfc))
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred_rfc))
    if genplot:
        plot_confusion_matrix(lr_model,
                              x_test,
                              y_test,
                              values_format='d',
                              display_labels=['Nondemented', 'Demented'])
        plt.title("Logistic Regression")
        plt.savefig("images/after_opt/LogisticRegression.png")
    return lr_model, lgr_fpr, lgr_tpr



def log_regWithGridView(x_train, y_train):
    print("-- Execute LR with Grid View")
    tic = time.perf_counter()
    # Create the param grid
    param_grid = {'penalty': ['l1', 'l2'],
                  'C': [0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 100, 1000]}

    optimal_params = GridSearchCV(LogisticRegression(),
                                  param_grid,
                                  cv=5,  # we are taking 5-fold as in k-fold cross validation
                                  scoring='accuracy',  # try the other scoring if have time
                                  verbose=0,
                                  n_jobs=-1)

    optimal_params.fit(x_train, y_train)
    optimal_params.fit(x_train, y_train)
    toc = time.perf_counter()
    return optimal_params, toc-tic
