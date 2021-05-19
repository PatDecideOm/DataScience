import pandas as pd
import numpy as np
import csv

print('numpy:        %s' % np.__version__)
print('pandas:       %s' % pd.__version__)

print('--- DEB ---')

dataset = pd.read_csv('./trans_invoice.csv', sep=";")

print(dataset.shape)

data = []

for i, trans in dataset.iterrows():

    items = trans['items']
    id = str(trans['id'])

    items = items.replace('[', '')
    items = items.replace(']', '')
    items = items.replace('"', '')

    for product in items.split(','):
        data.append([id, product])

df = pd.DataFrame(data, columns= ['id', 'product'])

print(df.shape)

df.to_csv('./transactions.csv',header=True, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)

print('--- FIN ---')