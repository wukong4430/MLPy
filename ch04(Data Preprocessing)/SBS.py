# -*- coding: utf-8 -*-
# @Author: kicc
# @Date:   2018-02-10 12:19:51
# @Last Modified by:   kicc
# @Last Modified time: 2018-02-10 13:23:48

from sklearn.base import clone
from itertools import combinations
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
# print('Class label :', np.unique(df_wine['Class label']))
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)


class SBS():
    """寻找最合适的features个数。
       estimator: classifier
       k_features: desired number of feature
       scoring: some metric to judge the measurment
       test_size: used in split

       从全部到features到k_features个数的features过程中，
       也就是下面plot绘制的横坐标X，对应的特征是已经经过筛选的，都是最优的。
    """

    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = estimator
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))  # like (0,1,2,3,4,5,6,7)
        self.subsets_ = [self.indices_]  # like [(0,1,2,3,4,5,6,7)]
        # 选择了所有的特征进行计算
        score = self._calc_score(
            X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            # return the index of the Maximum.
            best = np.argmax(scores)
            self.indices_ = subsets[best]  # 将最好的score对应的这些features选出来
            # 添加到self.subsets_中的都是n个features的最好的那个选择。
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 选择某几列(indices)进行fit，这里的indices就代表了features
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

# selecting features
sbs = SBS(estimator=knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting   performance of fffeature subts
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
# plt.show()

k5 = list(sbs.subsets_[8])  # k_features = 5
k6 = list(sbs.subsets_[7])  # k_features = 6
print(df_wine.columns[1:][k6])


# 证明确实提高的测试集的准确率 并且 减少了overfitting.
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))


knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))
