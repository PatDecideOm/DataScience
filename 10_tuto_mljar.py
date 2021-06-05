import pandas as pd
# scikit learn utilites
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# mljar-supervised package
from supervised.automl import AutoML

print('--- DEB ---')

dataset = pd.read_csv('C:/applis/kaggle/tabular-playground-series-jun-2021/train_cluster.csv', sep=';')
unseen = pd.read_csv('C:/applis/kaggle/tabular-playground-series-jun-2021/test_cluster.csv', sep=';')

print(dataset.shape)
print(unseen.shape)

y = dataset.target
X = dataset.drop('target', axis=1)

# Suppression ID
X = X.drop('id', axis=1)
id = unseen['id']
unseen = unseen.drop('id', axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.90, random_state=2103)
X_train = X
y_train = y

automl = AutoML(mode='Perform',
                ml_task='multiclass_classification',
                algorithms=('CatBoost','Neural Network'),
                validation_strategy='auto',
                total_time_limit=24 * 3600
                )

automl.fit(X_train, y_train)

# compute the accuracy on test data
predictions = automl.predict_all(X_validation)
print(predictions.head())
print("Test accuracy:", accuracy_score(y_validation, predictions["label"]))

# Compute prediction on test data
predictions = automl.predict_all(unseen)

submission = pd.DataFrame({'id': id,
                           'Class_1': predictions["prediction_Class_1"],
                           'Class_2': predictions["prediction_Class_2"],
                           'Class_3': predictions["prediction_Class_3"],
                           'Class_4': predictions["prediction_Class_4"],
                           'Class_5': predictions["prediction_Class_5"],
                           'Class_6': predictions["prediction_Class_6"],
                           'Class_7': predictions["prediction_Class_7"],
                           'Class_8': predictions["prediction_Class_8"],
                           'Class_9': predictions["prediction_Class_9"]})

submission.to_csv('C:/applis/kaggle/tabular-playground-series-jun-2021/submission_mljar.csv',
                    columns=['id', 'Class_1', 'Class_2', 'Class_3',
                             'Class_4', 'Class_5', 'Class_6',
                             'Class_7', 'Class_8',
                             'Class_9'], header=True, index=False)

print('--- FIN ---')