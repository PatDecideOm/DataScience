import pycaret
import pandas as pd
import csv

print(pycaret.__version__)

data = pd.read_csv('./train.csv')

from pycaret.arules import *
exp = setup(data = data, transaction_id = 'id', item_id = 'product')

rule1 = create_model(metric='confidence', threshold=0.75, min_support=0.01)

rule1["antecedent_len"] = rule1["antecedents"].apply(lambda x: len(x))
rule1["consequent_len"] = rule1["consequents"].apply(lambda x: len(x))

rule1["antecedents"] = rule1["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rule1["consequents"] = rule1["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

rule1.to_csv('./rules_caret.csv',
             header=True,
             index=False,
             encoding="utf-8",
             quoting=csv.QUOTE_NONNUMERIC,
             sep=';')