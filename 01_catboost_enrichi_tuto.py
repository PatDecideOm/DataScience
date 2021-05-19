# Import libraries
import pandas as pd
import numpy as np
import catboost
from catboost import CatBoostClassifier
import xgboost as xgb
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print('Scikit Learn: %s' % sklearn.__version__)
print('Catboost:     %s' % catboost.__version__)
print('xgboost:      %s' % xgb.__version__)
print('numpy:        %s' % np.__version__)
print('numpy:        %s' % pd.__version__)

print('--- DEB ---')

dataset = pd.read_csv('c:/applis/kaggle/tabular-playground-series-mar-2021/train_enrichi.csv')
unseen = pd.read_csv('c:/applis/kaggle/tabular-playground-series-mar-2021/test_enrichi.csv')

print(dataset.shape)
print(unseen.shape)

dataset = dataset[0:300000]
unseen = unseen[0:200000]

y = dataset.target
X = dataset.drop('target', axis=1)

columns = X.columns[1:]
print('Columns: %s' % columns)

cat_features = columns[:19]
print('Features: %s' % cat_features)

num_features = columns[20:]
print('Numerics: %s' % num_features)

train_test = pd.concat([X, unseen], ignore_index=True)

for feature in cat_features:
    le = LabelEncoder()
    le.fit(train_test[feature])
    X[feature] = le.transform(X[feature])
    unseen[feature] = le.transform(unseen[feature])

'''
for feature in num_features:
    X[feature] = (X[feature] - X[feature].mean(0)) / X[feature].std(0)
    unseen[feature] = (unseen[feature] - unseen[feature].mean(0)) / unseen[feature].std(0)
'''

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.80, random_state=2021)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=5000,
                           learning_rate=0.01,
                           depth=8,
                           #loss_function='CrossEntropy',
                           loss_function='Logloss',
                           #custom_loss=['AUC', 'Accuracy'],
                           task_type='GPU',
                           ignored_features=['id']
                           )

'''
grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}

grid_search_result = model.grid_search(grid,
                                       X=X_train,
                                       y=y_train,
                                       plot=False)
'''

# Fit model
model.fit(X_train, 
          y_train, 
          cat_features,
          eval_set=(X_validation, y_validation),
          verbose=250,
          plot=False)

print('Model is fitted : ' + str(model.is_fitted()))
print('Model params :')
print(model.get_params())

preds = model.predict(unseen)
probs= model.predict_proba(unseen)

unseen['target'] = probs[:,1]

submission = unseen[['id', 'target']]

print(submission.head())

submission.to_csv('c:/applis/kaggle/tabular-playground-series-mar-2021/submission_cat_enrichi.csv',
                    columns=['id', 'target'], header=True, index=False)

'''
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.010,
    "max_depth": 8,
    "n_jobs": 2,
    "seed": 2021,
    'tree_method': "hist"
}

train_df = xgb.DMatrix(X_train, label=y_train)
val_df = xgb.DMatrix(X_validation, label=y_validation)

model = xgb.train(xgb_params, train_df, 500)
temp_target = model.predict(val_df)

best_preds = [0 if line < 0.5 else 1 for line in temp_target]

from sklearn.metrics import precision_score

print(precision_score(best_preds, y_validation, average='macro'))

df_unseen = xgb.DMatrix(unseen)

# preds = model.predict(df_unseen)
'''

print('--- FIN ---')

