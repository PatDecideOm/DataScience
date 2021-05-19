import tablib
import tablib.core

print(tablib.__version__)

with open('train.csv', 'r') as fh:
    imported_data = tablib.Dataset().load(fh)

