import pandas as pd
# scikit learn utilites
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# mljar-supervised package
from supervised.automl import AutoML

print('--- DEB ---')

dataset = pd.read_csv('/home/patrice/kaggle/tabular-playground-series-jun-2021/train_cluster.csv', sep=';')
unseen = pd.read_csv('/home/patrice/kaggle/tabular-playground-series-jun-2021/test_cluster.csv', sep=';')

dataset=dataset[0:1000]
unseen=unseen[0:1000]

print(dataset.shape)
print(unseen.shape)

y = dataset.target
X = dataset.drop('target', axis=1)

# Suppression ID
X = X.drop('id', axis=1)
id = unseen['id']
unseen = unseen.drop('id', axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.90, random_state=2103)