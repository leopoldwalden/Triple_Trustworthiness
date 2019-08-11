import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt
from prepare import *
from sklearn.metrics import accuracy_score
import tqdm
from tqdm import trange

""" HELPER FUNCTION: GET ERROR RATE ========================================="""


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""


def print_error_rate(err):
    print
    'Error rate: Training: %.4f - Test: %.4f' % err


""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""


def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


""" ADABOOST IMPLEMENTATION ================================================="""


def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in trange(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # print(pred_train_i.shape)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x == 1 else -1 for x in miss]
        # Error
        err_m = np.dot(w, miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]
    model = clf
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return pred_train, Y_train, \
           pred_test, Y_test, model


""" PLOT FUNCTION ==========================================================="""


def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 450, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')


""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':

    # Read data
    # x, y = make_hastie_10_2()
    # df = pd.DataFrame(x)
    # df['Y'] = y

    path = "./data/low_encoded_data.csv"
    df = pd.read_csv(path)
    # new_df = pd.DataFrame(columns="")
    x_list, y_list, i2l = preprocess(df)

    x, y = np_array_data(x_list, y_list)
    # new_df = zip_data(x_list, y_list)

    # print(new_df.shape)
    # X = np.array(x_list)
    # Y = np.array(y_list)



    # Split into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
    # X_train, Y_train = train[:,0], train[:,1:]
    # X_test, Y_test = test[:,0], test[:,1:]

    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth=5, random_state=1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)

    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    # x_range = trange(10, 410, 10)
    # for i in x_range:
    pred_train, Y_train, pred_test, Y_test, clf = adaboost_clf(Y_train, X_train, Y_test, X_test, 400, clf_tree)
        # er_train.append(er_i[0])
        # er_test.append(er_i[1])
    joblib.dump(clf,"./model/adaboost.model")
    pred_test = clf.predict(X_test)
    # Compare error rate vs number of iterations
    # plot_error_rate(er_train, er_test)
    # [print(tup) for tup in zip(Y_test,pred_test)]
    accuracy = accuracy_score(Y_test, pred_test)
    # print(er_train.pop())
    # print(er_test.pop())
    print("accuracy:", accuracy)