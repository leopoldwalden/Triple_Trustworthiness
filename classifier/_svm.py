import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import joblib
from prepare import *

def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='linear', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf', 'linear', 'poly'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print('精度为%s' % score)
    return clf

if __name__ == '__main__':

    path = "./data/low_encoded_data.csv"
    df = pd.read_csv(path)
    x_list, y_list, i2l = preprocess(df)
    x, y = np_array_data(x_list, y_list)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

    svc = SVC(kernel='linear', gamma=0.001953125, C=0.03125, class_weight='balanced', )
    model = svc.fit(X_train,Y_train)
    pred_test = model.predict(X_test)
    accuracy = accuracy_score(pred_test, Y_test)
    print("accuracy:", accuracy)
    """
        accuracy: 83.71%
    """


    # model = svm_c(X_train, X_test, Y_train, Y_test)
    joblib.dump(model, "./model/svm.model")
    # print(model.best_params_)
    """
    10-20： {'C': 0.03125, 'gamma': 0.001953125, 'kernel': 'rbf'}
    50：    {'C': 0.03125, 'gamma': 0.001953125, 'kernel': 'linear'}
    
    """

    # pred_test = model(X_test)
    # accuracy = accuracy_score(pred_test, Y_test)
    # print("accuracy:", accuracy)

