import _xgboost
from numpy import loadtxt
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from gen_features import *
# from sklearn.externals import joblib
import joblib
from macro import ROOT_PATH

# from prepare import *
# ROOT_PATH = "/home/jilei/Desktop/PycharmProjects/Triple_Trustworthiness"


# 载入数据集
# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# # split data into X and y
# X = dataset[:, 0:8]
# Y = dataset[:, 8]
def main():
    # path = "./data/low_encoded_data.csv"
    # # path = "./data/dev_data.csv"
    # df = pd.read_csv(path)
    features = sample2feature()
    # x_list, y_list, i2l = preprocess(df)
    X = features[:,0]
    print("X.shape:",X.shape)
    Y = features[:,0:]
    print("Y.shape:",Y.shape)

    # print(Y)
    # dataset = zip_data(x_list, y_list)
    # X = dataset.x.tolist()
    # Y = dataset.y.tolist()

    # 把数据集拆分成训练集和测试集
    seed = 7
    test_size = 0.33
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed )

    # 拟合XGBoost模型
    model = XGBClassifier(gamma= 0.001953125, max_depth= 4)
    model.fit(x_train, y_train)

    # grid search
    # gamma_range = np.logspace(-9, 3, 13, base=2)
    # max_depth_range = np.arange(1,11,1)
    # paradict = [{"booster": ["gblinear", "gbtree"], "gamma": gamma_range, "max_depth": max_depth_range}]
    # grid = GridSearchCV(estimator = model, param_grid = paradict, cv=3, n_jobs=-1)
    # grid_model = grid.fit(X,Y)
    # print(grid_model.best_params_)
    """
    {'booster': 'gblinear', 'gamma': 0.001953125, 'max_depth': 1}
    
    """

    # save model
    joblib.dump(model, ROOT_PATH + "/model/xgboost.model")
    # model = joblib.load("./model/xgboost.model")

    # # 对测试集做预测
    predictions = model.predict(x_test)
    # zipped = zip(y_test,predictions)
    # [print(i2l[tup[0]],i2l[tup[1]]) for tup in zipped if tup[0]!=tup[1]]
    predictions = [round(value) for value in predictions]
    #
    # # 评估预测结果
    accuracy = accuracy_score(y_test, predictions)
    # f1 = f1_score(y_test, predictions)
    print("test")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # # print("F1: %.2f%%" % (f1 * 100.0))


if __name__=='__main__':
    main()

"""
Baseline Accuracy: 87.46%
"""