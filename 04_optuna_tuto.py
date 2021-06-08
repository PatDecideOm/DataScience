import warnings
warnings.filterwarnings('ignore')
import catboost as cb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import matplotlib

import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

print('--- DEB ---')

print('Scikit Learn: %s' % sklearn.__version__)
print('Catboost:     %s' % cb.__version__)
print('numpy:        %s' % np.__version__)
print('pandas:       %s' % pd.__version__)

dataset = pd.read_csv('C:/applis/kaggle/tabular-playground-series-jun-2021/train.csv', sep=',')
unseen = pd.read_csv('C:/applis/kaggle/tabular-playground-series-jun-2021/test.csv', sep=',')

# dataset=dataset[0:500]
unseen=unseen[0:1000]

print(dataset.shape)
print(unseen.shape)

''' Parameters to optimize
    'n_estimators' / 'iterations' : ok
    'depth' : ok
    'learning_rate' : ok
    'colsample_bylevel' 
    'bagging_temperature' - ko
    'l2_leaf_reg'
    'bootstrap_type' : ok
    'min_child_samples'
    'subsample' : ok
'''

y = dataset.target
X = dataset.drop('target', axis=1)

# Suppression ID
X = X.drop('id', axis=1)
id = unseen['id']
unseen = unseen.drop('id', axis=1)

columns = X.columns[0:75]
cat_features = columns
print('Features: %s' % cat_features)

def objective(trial):

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.85, random_state=2103)

    # Parameters
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.25),
        'l2_leaf_reg': trial.suggest_uniform('l2_leaf_reg', 0, 10),
    }

    # Learning
    gbm = cb.CatBoostClassifier(
        loss_function='MultiClass',
        custom_loss=['Accuracy'],
        task_type="GPU",
        devices='0:1',
        random_seed=210369,
        bootstrap_type='Bayesian',
        # subsample=0.66, -- 'Poisson'
        **params
    )

    gbm.fit(X_train, y_train, cat_features, eval_set=[(X_valid, y_valid)], verbose=0, early_stopping_rounds=100)

    # compute the accuracy on test data
    predictions = gbm.predict_proba(X_valid)
    y_pred = gbm.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    print(gbm.get_best_score())

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, timeout=3600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_optimization_history(study)
