# -*- coding: utf-8 -*-
# @Author: kicc
# @Date:   2018-02-06 14:36:03
# @Last Modified by:   kicc
# @Last Modified time: 2018-02-10 12:19:40

import numpy as np
import pandas as pd
from io import StringIO


def pandas_csv():
    csv_data = '''A,B,C,D
            1.0,2.0,3.0,4.0
            5.0,6.0,,8.0
            0.0,11.0,,13.0
    '''
    # DataFrame
    df = pd.read_csv(StringIO(csv_data))
    print(df)

    print(df.isnull().sum())

    # transform df to numpy-value
    print(type(df.values))

    ar = np.array([[1, 3, 4],
                   [5, 67, 2],
                   [4, 2, 55]])

    print(type(ar))

    # 没有丢失数据的行\列
    print(df.dropna())  # 行
    print(df.dropna(axis=1))  # 列
    # several useful parameters
    print(df.dropna(how='all'))
    print(df.dropna(thresh=4))
    print(df.dropna(subset=['C']))


def imputing_missing_values():
    csv_data = '''A,B,C,D
            1.0,2.0,3.0,4.0
            5.0,6.0,,8.0
            0.0,11.0,12.0,
    '''
    # DataFrame
    df = pd.read_csv(StringIO(csv_data))

    from sklearn.preprocessing import Imputer
    # 用同一个feature的其他数值的mean代替missing data
    # axis=0是column均值，axis=1是row均值
    # strategy 还可以是 most_frequent, median
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # imr = imr.fit(df)
    # imputed_data = imr.transform(df.values)
    # fit_transform() 可以代替fit() & transform()
    imputed_data = imr.fit_transform(df.values)
    print(imputed_data)


def handling_categrical_data():
    # DataFrame contains various categories.
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])
    df.columns = ['color', 'size', 'price', 'classlabel']
    print(df)
    print('-' * 30)
    # mapping_ordinal_features
    # 将文字描述信息与数字映射起来
    size_mapping = {
        'XL': 3,
        'L': 2,
        'M': 1
    }
    df['size'] = df['size'].map(size_mapping)
    print(df)

    # encoding class labels
    class_mapping = {label: idx for idx,
                     label in enumerate(np.unique(df[['classlabel']]))}
    print(class_mapping)

    print('-' * 30)
    df['classlabel'] = df['classlabel'].map(class_mapping)
    print(df)

    color_mapping = {
        'green': 1,
        'red': 2,
        'blue': 3
    }
    df['color'] = df['color'].map(color_mapping)
    print('-' * 30)
    print(df)

    # 反转key，value
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    print(inv_class_mapping)
    df['classlabel'] = df['classlabel'].map(inv_class_mapping)
    print('-' * 30)
    print(df)

    # An alternative implemention in sklearn.
    from sklearn.preprocessing import LabelEncoder
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    print(df['classlabel'].values)
    print(y)
    print(class_le.inverse_transform(y))

    # 数据分析比较常规(有用的步骤)
    # 1.提取values
    X = df[['color', 'size', 'price']].values
    print('The ordinary X:')
    print(X)
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    print('After encoding the 1st column:')
    print(X)
    print('After encoding the 2nd column:')
    X[:, 1] = color_le.fit_transform(X[:, 1])
    print(X)
    print('After encoding the 3rd column:')
    X[:, 2] = color_le.fit_transform(X[:, 2])
    print(X)

    # using one-hot encoding.
    # 很类似与将连续值转化成离散值的做法。
    # 就是用0-1 binary.
    X = df[['color', 'size', 'price']].values
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(categorical_features=[0])
    X_array = ohe.fit_transform(X).toarray()
    print('Encoding with OneHotEncoder(column 0):')
    print(X_array)
    ohe = OneHotEncoder(categorical_features=[0, 1])
    X_array = ohe.fit_transform(X).toarray()
    print('Encoding with OneHotEncoder(column 0&1):')
    print(X_array)

    # more convenient way to alternate one-hot encoding.
    # get_dummies() will only convert string columns.
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])
    df.columns = ['color', 'size', 'price', 'classlabel']
    dummy_features = pd.get_dummies(df[['price', 'color', 'size']])

    print('Use get_dummies() ')
    print(dummy_features)


def partitioning_training_test():
    """Use a new dataset, the Wine dataset.

    """
    df_wine = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']
    print('Class label :', np.unique(df_wine['Class label']))
    from sklearn.cross_validation import train_test_split
    X, y = df.wine.iloc[:, 1:].values, df.wine[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # 规范化
    # Xi-norm = (Xi - Xmin) / (Xmax - Xmin)
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.fit_transform(X_test)

    # 标准化
    # Xi-std = (Xi - Xmean) / (Xstandard deviation)
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)


def regularization():
    """Use L1 and L2 regularization.

    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import train_test_split
    df_wine = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']
    print('Class label :', np.unique(df_wine['Class label']))
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)

    # Use l1 regularization in LogisticRegression.
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=0.1)
    lr.fit(X_train_std, y_train)
    print('Training accuracy :', lr.score(X_train_std, y_train))
    print('Test accuracy :', lr.score(X_test_std, y_test))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.subplot(121)  # 1 row, 2 cols, 1 index
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow',
              'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
    weights, params = [], []
    for c in np.arange(-4, 6, dtype=float):
        lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)
    weights = np.array(weights)
    # weights.shape[1] = #features
    # print('Weights :', weights, 'Weights shape[1] :', weights.shape[1])
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column],
                 label=df_wine.columns[column + 1],
                 color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.xlabel('C')
    plt.ylabel('weight coefficient')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='best',
              bbox_to_anchor=(1.38, 1.03),
              ncol=1, fancybox=True)
    plt.show()


def feature_selection():
    """Reduce the complexity of the model.

    There are two techniques: feature selection and feature extraction.
    extraction: Derive information
                from the feature set to construct a new feature subspace.(Next chapter)
    """
    pass


def main():
    # imputing_missing_values()
    # handling_categrical_data()
    # partitioning_training_test()
    regularization()


if __name__ == '__main__':
    main()
