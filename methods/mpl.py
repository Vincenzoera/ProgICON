import warnings
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import zero_one_loss, classification_report, confusion_matrix, roc_curve, auc, \
    plot_confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
import time

warnings.filterwarnings("ignore")
def mlpClassifier(x_train, x_test, y_train, y_test, y, infoPrint, genplot):
    # Multi Layer Perceptron
    if infoPrint:
        print("\n-- MULTI LAYER PERCEPTRON")

    # Definizione e classificazione della Rete Neurale
    mlp = MLPClassifier(activation='relu')
    mlp.fit(x_train, y_train)
    y_pred_mlp = mlp.predict(x_test)
    ncv_score_mlp = mlp.score(x_test, y_test)
    score_train_mlp = mlp.score(x_train, y_train)
    score_mlp = cross_val_score(mlp, x_train, y_train, cv=2)
    mlp_fpr, mlp_tpr, thr = roc_curve(y_test, y_pred_mlp)
    accuracy = auc(mlp_fpr, mlp_tpr)
    if infoPrint:
        print("Train score: ", score_train_mlp, "\nTest score: ", ncv_score_mlp)
        print("Cross Validated Score: ", score_mlp.mean())
        print("Standard Deviation: ", score_mlp.std())
        print("Variance: ", np.var(score_mlp))
        print("0-1 Loss: ", zero_one_loss(y_test, y_pred_mlp))
        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred_mlp))

    if genplot:
        plot_confusion_matrix(mlp,
                              x_test,
                              y_test,
                              values_format='d',
                              display_labels=['Nondemented', 'Demented'])
        plt.title("Multi Layer Perceptron")
        plt.savefig("images/after_opt/MultiLayerPerceptron.png")

    return mlp, mlp_fpr, mlp_tpr

def mlpWithGridView(x_train, y_train):
    print("-- Execute MPL with Grid View")
    tic = time.perf_counter()
    param_grid = {'activation': ['relu','logistic','tanh','identity']}
    search = GridSearchCV(MLPClassifier(max_iter=200), param_grid, cv=5, refit=True, error_score=0, n_jobs=-1)
    search.fit(x_train, y_train)
    toc = time.perf_counter()
    return search, toc-tic
