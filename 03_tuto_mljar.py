import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML

print('--- DEB ---')

dataset = pd.read_csv('/home/patrice/kaggle/tabular-playground-series-mar-2021/train.csv')
unseen = pd.read_csv('/home/patrice/kaggle/tabular-playground-series-mar-2021/test.csv')

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

'''
for feature in cat_features:
    le = LabelEncoder()
    le.fit(train_test[feature])
    X[feature] = le.transform(X[feature])
    unseen[feature] = le.transform(unseen[feature])
'''

'''
for feature in num_features:
    X[feature] = (X[feature] - X[feature].mean(0)) / X[feature].std(0)
    unseen[feature] = (unseen[feature] - unseen[feature].mean(0)) / unseen[feature].std(0)
'''

# Suppression ID
X = X.drop('id', axis=1)
ID = unseen['id']
unseen = unseen.drop('id', axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.90, random_state=0)
X_train = X
y_train = y

automl = AutoML(mode='Compete', ml_task='binary_classification',
                algorithms=('CatBoost','LightGBM'),
                validation_strategy='auto',
                total_time_limit=14400
                )
automl.fit(X_train, y_train)

# predictions = automl.predict(unseen) -- to obtain 0 and 1

predictions = automl.predict_proba(unseen)
preds = predictions[:, 1]

submission = pd.DataFrame({'id': ID, 'target': preds})
submission.to_csv('/home/patrice/kaggle/tabular-playground-series-mar-2021/submission_mljar.csv',
                    columns=['id', 'target'], header=True, index=False)

print('--- FIN ---')
