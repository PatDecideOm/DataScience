from pyarrow import csv
import pandas as pd
import pyarrow as pa

fn = 'C:/applis/kaggle/tabular-playground-series-may-2021/train.csv'

table = pa.csv.read_csv(fn)

df = table.to_pandas()
