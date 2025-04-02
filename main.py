import pickle
import warnings
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from methods.KNN import knnClassifier, kncWithGridView
from methods.RandomForestClassifer import rfClassifier, rfcWithGridView
from methods.decision_tree import dtClassifier, dtcWithGridView
from methods.kmeans import kmns, kmns_k_search
from methods.logisticRegression import log_reg, log_regWithGridView
from methods.mpl import mlpClassifier, mlpWithGridView
from utils.utils import loadDataSet
import seaborn as sns
from tensorflow import keras
from sklearn import metrics

warnings.filterwarnings("ignore")

dct_filename = 'models/dct_pkl.pkl'
knc_filename = 'models/knc_model.pkl'
rfc_filename = 'models/rfc_model.pkl'
mlp_filename = 'models/mlp_model.pkl'
lr_filename = 'models/lr_model.pkl'
kms_filename = 'models/kmeans_model.pkl'

genplot = True  # impostare su True per generare e salvare i grafici
X_train, X_test, y_train, y_test, x, y, data = loadDataSet(genplot)
df_ytrain = pd.DataFrame(y_train)
df_ytest = pd.DataFrame(y_test)
scaler = StandardScaler().fit(X_train)
# scaler = MinMaxScaler().fit(X_trainval)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


def notsup():
    tic = time.perf_counter()
    kms, kms_frp, kms_tpr = kmns(X_train_scaled, X_test_scaled, y_train, y_test, True)
    pickle.dump(kms.fit(X_train, y_train), open(kms_filename, 'wb'))
    toc = time.perf_counter()
    print(f"\n-- Learning completed in {toc - tic:0.4f} seconds")
    pickle.dump(kms.fit(X_train, y_train), open(kms_filename, 'wb'))

def executeClassifier():
    tic = time.perf_counter()
    dct, dct_fpr, dct_tpr = dtClassifier(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    knn, knn_fpr, knn_tpr = knnClassifier(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    logr, lgr_fpr, lgr_tpr = log_reg(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    rfc, rfc_fpr, rfc_tpr = rfClassifier(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    mlp, mlp_frp, mlp_tpr = mlpClassifier(X_train_scaled, X_test_scaled, y_train, y_test, y, True, genplot)
    kms, kms_frp, kms_tpr = kmns(X_train_scaled, X_test_scaled, y_train, y_test, False)
    toc = time.perf_counter()
    print(f"\n-- Learning completed in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    '''export dei classificatori addestrati'''
    pickle.dump(dct.fit(X_train, y_train), open(dct_filename, 'wb'))
    pickle.dump(knn.fit(X_train, y_train), open(knc_filename, 'wb'))
    pickle.dump(rfc.fit(X_train, y_train), open(rfc_filename, 'wb'))
    pickle.dump(mlp.fit(X_train, y_train), open(mlp_filename, 'wb'))
    pickle.dump(logr.fit(X_train, y_train), open(lr_filename, 'wb'))

    toc = time.perf_counter()
    print(f"-- Models saved in {toc - tic:0.4f} seconds\n")

    if genplot:
        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(lgr_fpr, lgr_tpr, marker='.', label='Logistic Regression')
        plt.plot(rfc_fpr, rfc_tpr, linestyle=':', label='Random Forest')
        plt.plot(dct_fpr, dct_tpr, linestyle='-.', label='Decision Tree')
        plt.plot(knn_fpr, knn_tpr, linestyle='-.', label='K-Neighbors')
        plt.plot(mlp_frp, mlp_tpr, linestyle='-.', label='Multi Layer Perceptron')
        plt.plot(kms_frp, kms_tpr, linestyle='-.', label='K-Means')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # plt.show()
        plt.savefig("images/comparazione.png")


def gridSearch():
    dtc_gd, dtc_time = dtcWithGridView(X_train_scaled, y_train)  # Decision Tree Classifer
    knn_gd, knc_time = kncWithGridView(X_train_scaled, y_train)  # Multi Layer Perceptron
    rfc_gd, rfc_time = rfcWithGridView(X_train_scaled, y_train)  # K-Neighbors
    mlp_gd, mlp_time = mlpWithGridView(X_train_scaled, y_train)  # Multi Layer Perceptron
    lr_gd, lr_time = log_regWithGridView(X_train_scaled, y_train) #K-Means
    score_k, kmns_time = kmns_k_search(x, genplot)
    print("\nBEST PARAMS:")
    print(f"- DTC: {dtc_gd.best_params_} - {dtc_time:0.4f} seconds")
    print(f"- KNN: {knn_gd.best_params_} - {knc_time:0.4f} seconds")
    print(f"- RFC: {rfc_gd.best_params_} - {rfc_time:0.4f} seconds")
    print(f"- MLP: {mlp_gd.best_params_} - {mlp_time:0.4f} seconds")
    print(f"- LR: {lr_gd.best_params_} - {lr_time:0.4f} seconds")
    print(f"- kMeans: {score_k:0.4f} - {kmns_time:0.4f} seconds")
    print(f"-- Fine optimization end in {dtc_time + knc_time + rfc_time + mlp_time + lr_time + kmns_time:0.4f} seconds")

#Procedure of prediction
def prediction():

    tic = time.perf_counter()
    dct_load = pickle.load(open(dct_filename, 'rb'))
    knn_load = pickle.load(open(knc_filename, 'rb'))
    rfc_load = pickle.load(open(rfc_filename, 'rb'))
    mlp_load = pickle.load(open(mlp_filename, 'rb'))
    lr_load = pickle.load(open(lr_filename, 'rb'))
    kms_load = pickle.load(open(kms_filename, 'rb'))
    toc = time.perf_counter()
    print(f"\n-- Loaded trainend models in {toc - tic:0.4f} seconds")

    #ARRAY STATICO che comprende i dati input delle feature rilevanti per la predizione della demenza senile.

    # 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'
    xx = [[0, 87, 14, 2, 27, 1987, 0.696, 0.883]]  # real: nondemented 1
    xy = [[1, 80, 12, 2.0, 22.0, 1698, 0.701, 1.034]]  # real: demented 4
    xz = [[1, 76, 12, 2.0, 28.0, 1738, 0.713, 1.010]]  # real: demented 3
    zz = [[1, 99, 8, 2.0, 24.0, 1679, 0.800, 0.987]]  # caso inventato
   

    test = xx #Test scelto per l'esempio.
    tic = time.perf_counter()
    print("\nTEST: ", test[0], "\n")
    pred1 = dct_load.predict(test)
    pred2 = knn_load.predict(test)
    pred3 = rfc_load.predict(test)
    pred4 = mlp_load.predict(test)
    pred5 = lr_load.predict(test)
    pred6 = kms_load.predict(test)

    # PREDIZIONE , La maggioranza decreterà l'esito finale, quindi demented o non demented.
    
    d = {'DCT': [pred1[0]], 'KN': [pred2[0]], 'RaF': [pred3[0]], 'MLP': [pred4[0]], 'LR': [pred5[0]], 'KMS': [pred6[0]]}
    prev = pd.DataFrame(data=d)

    demented = "demented" if prev.mode(1).loc[0].values[0] == 1 else "nondemented"
    with pd.option_context('display.max_rows', 2, 'display.max_columns', 6):
        print(prev)
    print("The most predicted disease is", demented)

    toc = time.perf_counter()
    print(f"\n-- Predicted in {toc - tic:0.4f} seconds\n")


print("Alzheimer prediction\n")

menu_options = {
    1: 'Learning Supervised',
    2: 'Learning Unsupervised',
    3: 'Optimize parameters',
    4: 'Run test',
    5: 'Exit',
}


def print_menu():
    for key in menu_options.keys():
        print(key, '--', menu_options[key])


if __name__ == '__main__':
    while (True):
        print_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        if option == 1:
            executeClassifier()
        elif option == 2:
            notsup()
        elif option == 3:
            gridSearch()
        elif option == 4:
            prediction()
        elif option == 5:
            print('Exit')
            exit()
        else:
            print('Invalid option. Please enter a number between 1 and 4.')
