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

for feature in num_features:
    X[feature] = (X[feature] - X[feature].mean(0)) / X[feature].std(0)
    unseen[feature] = (unseen[feature] - unseen[feature].mean(0)) / unseen[feature].std(0)

# Suppression ID
X = X.drop('id', axis=1)
ID = unseen['id']
unseen = unseen.drop('id', axis=1)

# X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.80, random_state=2021)

def objective(trial, data=X, target=y):

    X_train, X_validation, y_train, y_valid = train_test_split(X, y, train_size=0.85, random_state=0)

    params = {
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.02, 0.05, 0.08]),
        'n_estimators': trial.suggest_int('n_estimators', 2000, 4000),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.1, 0.8),
        'random_seed': 69,
        #'task_type': 'CPU',
        'task_type': 'GPU',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        #'bootstrap_type': 'Bernoulli'
        'bootstrap_type': 'Poisson'
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], early_stopping_rounds=300, verbose=False)
    y_pred = model.predict_proba(X_validation)[:, 1]
    roc_auc = roc_auc_score(y_valid, y_pred)

    return roc_auc

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 5)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print('Best value:', study.best_value)

paramsCB = study.best_trial.params
paramsCB['task_type'] = 'CPU'
paramsCB['loss_function'] = 'Logloss'
paramsCB['eval_metric'] = 'AUC'
paramsCB['random_seed'] = 69
paramsCB['bootstrap_type'] = 'Bernoulli'

from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=42)

predictions = np.zeros(len(unseen))

for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, X_validation = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_validation = y.iloc[trn_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**paramsCB)

    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False, early_stopping_rounds=300)

    predictions += model.predict_proba(unseen)[:, 1] / folds.n_splits

submission = pd.DataFrame({'id': ID, 'target': predictions})
submission.to_csv('/home/patrice/kaggle/tabular-playground-series-mar-2021/submission_cb.csv',
                    columns=['id', 'target'], header=True, index=False)

print('--- FIN ---')
