import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import catboost
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

print('Catboost: %s' % catboost.__version__)

print('--- DEB ---')

dataset = pd.read_csv('C:/applis/kaggle/tabular-playground-series-may-2021/train.csv')
unseen = pd.read_csv('C:/applis/kaggle/tabular-playground-series-may-2021/test.csv')

print(dataset.shape)
print(unseen.shape)

y = dataset.target
X = dataset.drop('target', axis=1)

columns = X.columns[1:]
print('Columns: %s' % columns)

cat_features = columns[:50]
print('Features: %s' % cat_features)

# Drop 'id'
X = X.drop('id', axis=1)
id = unseen['id']
unseen = unseen.drop('id', axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.85, random_state=2021)

print(X_train.head())

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2000,
                           learning_rate=0.1,
                           depth=10,
                           loss_function='MultiClass',
                           custom_loss=['Accuracy'],
                           task_type='GPU')

# Fit model
model.fit(X_train,
          y_train,
          cat_features,
          eval_set=(X_validation, y_validation),
          verbose=50,
          # plot=True,
          )

print('Model is fitted : ' + str(model.is_fitted()))
print('Model params :')
print(model.get_params())

preds = model.predict(unseen)
predictions = model.predict_proba(unseen)

submission = pd.DataFrame({'id': id,
                           'Class_1': predictions[:, 0],
                           'Class_2': predictions[:, 1],
                           'Class_3': predictions[:, 2],
                           'Class_4': predictions[:, 3]})

submission.to_csv('C:/applis/kaggle/tabular-playground-series-may-2021/submission_catboost.csv',
                    columns=['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4'], header=True, index=False)

print('--- FIN ---')