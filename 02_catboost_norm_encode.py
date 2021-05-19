# Import libraries
import pandas as pd
import numpy as np
import catboost
from catboost import CatBoostClassifier
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import optuna

import warnings
warnings.filterwarnings("ignore")

print('Scikit Learn: %s' % sklearn.__version__)
print('Catboost:     %s' % catboost.__version__)
print('numpy:        %s' % np.__version__)
print('pandas:       %s' % pd.__version__)

print('--- DEB ---')

dataset = pd.read_csv('/home/patrice/kaggle/tabular-playground-series-mar-2021/train.csv')
unseen = pd.read_csv('/home/patrice/kaggle/tabular-playground-series-mar-2021/test.csv')

print(dataset.shape)
print(unseen.shape)

dataset = dataset[0:300000]
unseen = unseen[0:200000]

y = dataset.target
# X = dataset.drop('target', axis=1)
X = dataset

columns = X.columns[1:]
print('Columns: %s' % columns)

cat_features = columns[:19]
print('Features: %s' % cat_features)

num_features = columns[20:-1]
print('Numerics: %s' % num_features)

train_test = pd.concat([X, unseen], ignore_index=True)

for feature in cat_features:
    le = LabelEncoder()
    le.fit(train_test[feature])
    X[feature] = le.transform(X[feature])
    unseen[feature] = le.transform(unseen[feature])

for feature in num_features:
    X[feature] = (X[feature] - X[feature].mean(0)) / X[feature].std(0)
    unseen[feature] = (unseen[feature] - unseen[feature].mean(0)) / unseen[feature].std(0)

X.to_csv('/home/patrice/kaggle/tabular-playground-series-mar-2021/train_mat.csv',
                    header=True, index=False)
unseen.to_csv('/home/patrice/kaggle/tabular-playground-series-mar-2021/test_mat.csv',
                    header=True, index=False)

print('--- FIN ---')
