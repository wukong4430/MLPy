# -*- coding: utf-8 -*-
# @Author: kicc
# @Date:   2018-01-29 17:00:09
# @Last Modified by:   kicc
# @Last Modified time: 2018-01-29 20:00:31

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.tree import export_graphviz


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])


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


def show_three_impurity():
    x = np.arange(0.0, 1.0, 0.01)

    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e * 0.5 if e else None for e in ent]
    err = [error(i) for i in x]

    fig = plt.figure()
    ax = plt.subplot(111)

    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                              ['Entropy', 'Entropy (scaled)', 'Gini Impurity',
                               'Misclassification Error'],
                              ['-', '-', '--', '-.'],
                              ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.15), ncol=3, fancybox=True, shadow=False)
    # draw two axis-parallel lines.
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')
    plt.show()


def decision_tree1():
    X_train_std, y_train, X_test_std, y_test = load_data()
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(
        criterion='entropy', max_depth=3, random_state=0)
    tree.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined,
                          classifier=tree, test_idx=range(105, 150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()

    export_graphviz(tree, out_file='tree.dot', feature_names=[
                    'petal length', 'peatl width'])


def random_forest():
    X_train_std, y_train, X_test_std, y_test = load_data()
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(
        criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    forest.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined,
                          classifier=forest, test_idx=range(105, 150))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='upper left')
    plt.show()


def KNN():
    X_train_std, y_train, X_test_std, y_test = load_data()
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined,
                          classifier=knn, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.xlabel('petal width [standardized]')
    plt.show()


def main():
    # decision_tree1()
    # random_forest()
    KNN()


if __name__ == '__main__':
    main()
