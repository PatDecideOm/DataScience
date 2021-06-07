import warnings
warnings.filterwarnings('ignore')
import catboost as cb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import optuna

print('--- DEB ---')

print('Scikit Learn: %s' % sklearn.__version__)
print('Catboost:     %s' % cb.__version__)
print('numpy:        %s' % np.__version__)
print('pandas:       %s' % pd.__version__)

dataset = pd.read_csv('C:/applis/kaggle/tabular-playground-series-jun-2021/train_cluster.csv', sep=';')
unseen = pd.read_csv('C:/applis/kaggle/tabular-playground-series-jun-2021/test_cluster.csv', sep=';')

# dataset=dataset[0:500]
unseen=unseen[0:1000]

print(dataset.shape)
print(unseen.shape)

y = dataset.target
X = dataset.drop('target', axis=1)

# Suppression ID
X = X.drop('id', axis=1)
X = X.drop('cluster', axis=1)
id = unseen['id']
unseen = unseen.drop('id', axis=1)

columns = X.columns[0:75]
cat_features = columns
print('Features: %s' % cat_features)

def objective(trial):

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.90, random_state=2103)

    # Parameters
    params = {
        'iterations' : trial.suggest_int('iterations', 50, 500),
        'depth' : trial.suggest_int('depth', 4, 12),
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.005, 0.25),
        'random_strength' :trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter'])
    }

    # Learning
    gbm = cb.CatBoostClassifier(
        loss_function='MultiClass',
        custom_loss=['Accuracy'],
        task_type="GPU",
        random_seed=210369,
        **params
    )

    gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0, early_stopping_rounds=100)

    # compute the accuracy on test data
    predictions = gbm.predict_proba(X_valid)
    y_pred = gbm.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    print(gbm.get_best_score())

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=3600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))