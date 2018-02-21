# -*- coding: utf-8 -*-
# @Author: kicc
# @Date:   2018-01-14 15:55:26
# @Last Modified by:   kicc
# @Last Modified time: 2018-01-29 16:57:57

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from matplotlib.colors import ListedColormap


def load_data():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # print('Class labels:', np.unique(y))

    from sklearn.cross_validation import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, y_train, X_test_std, y_test


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # xx1.ravel() makes matrix to array
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=55,
                    label='test set')


def svm_1():

    # large C means strict seperable; small C means more soft-margin
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    X_train_std, y_train, X_test_std, y_test = load_data()

    svm.fit(X_train_std, y_train)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined_std, y_combined,
                          classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [Standardize]')
    plt.ylabel('petal width [Standardize]')
    plt.legend(loc='upper left')
    plt.show()


def SGDClassifier():
    from sklearn.linear_model import SGDClassifier
    ppn = SGDClassifier(loss='perceptron')
    lr = SGDClassifier(loss='log')
    svm = SGDClassifier(loss='hinge')


def kernel_svm():
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)  # create a matrix 200×2
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)

    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
                c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
                c='r', marker='s', label='-1')
    plt.ylim(-3.0)
    plt.legend()
    plt.show()


def kernel_rbf():
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)  # create a matrix 200×2
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    # larger gamma means a softer decision boundary.
    #
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.show()


def kernel_rbf2():
    X_train_std, y_train, X_test_std, y_test = load_data()
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    svm = SVC(kernel='rbf', random_state=0, gamma=0.13, C=1.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined,
                          classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [Standardize]')
    plt.ylabel('petal width [standardize]')
    plt.legend(loc='upper left')
    plt.show()


def main():
    # svm_1()
    # kernel_svm()
    # kernel_rbf()
    kernel_rbf2()


if __name__ == '__main__':
    main()
