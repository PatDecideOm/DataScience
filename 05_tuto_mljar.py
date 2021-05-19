import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML

def display_missing(df):
    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

print('--- DEB ---')

dataset = pd.read_csv('/home/patrice/kaggle/tabular-playground-series-apr-2021/train.csv')
unseen = pd.read_csv('/home/patrice/kaggle/tabular-playground-series-apr-2021/test.csv')

print(dataset.shape)
print(unseen.shape)

dataset = dataset[0:100000]
unseen = unseen[0:100000]

# TRAIN
# Filling the missing values 'Age'
age_by_pclass_sex = dataset.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(dataset['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
dataset['Age'] = dataset.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

# Filling the missing values in Embarked with S
dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Filling the missing value 'Fare'
med_fare = dataset.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
dataset['Fare'] = dataset['Fare'].fillna(med_fare)

# TEST
# Filling the missing values 'Age'
age_by_pclass_sex = unseen.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(unseen['Age'].median()))

# Filling the missing values in Age with the medians of Sex and Pclass groups
unseen['Age'] = unseen.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

# Filling the missing values in Embarked with S
unseen['Embarked'] = unseen['Embarked'].fillna('S')

# Filling the missing value 'Fare'
med_fare = unseen.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
unseen['Fare'] = unseen['Fare'].fillna(med_fare)

y = dataset.Survived
X = dataset.drop('Survived', axis=1)

columns = X.columns[1:]
print('Columns: %s' % columns)

cat_features = [ columns[0], columns[1], columns[2], columns[4], columns[5],
                 columns[9] ]
print('Features: %s' % cat_features)

num_features = [ columns[3], columns[7] ]
print('Numerics: %s' % num_features)

display_missing(X)
display_missing(unseen)

# Drop columns with missing values
X = X.drop('Ticket', axis=1)
X = X.drop('Cabin', axis=1)

unseen = unseen.drop('Ticket', axis=1)
unseen = unseen.drop('Cabin', axis=1)

display_missing(X)
display_missing(unseen)

# Suppression ID
X = X.drop('PassengerId', axis=1)
ID = unseen['PassengerId']
unseen = unseen.drop('PassengerId', axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.90, random_state=0)

X_train = X
y_train = y

automl = AutoML(mode='Perform', ml_task='binary_classification',
                # algorithms=('CatBoost','LightGBM'),
                validation_strategy='auto',
                total_time_limit=14400
                )

automl.fit(X_train, y_train)

predictions = automl.predict(unseen) # to obtain 0 and 1

'''
predictions = automl.predict_proba(unseen)
preds = predictions[:, 1]
'''

submission = pd.DataFrame({'PassengerId': ID, 'Survived': predictions})
submission.to_csv('/home/patrice/kaggle/tabular-playground-series-apr-2021/submission_mljar.csv',
                    columns=['PassengerId', 'Survived'], header=True, index=False)

print('--- FIN ---')
