from pydruid.db import connect

conn = connect(host='localhost', port=8082, path='/druid/v2/sql/', scheme='http')

curs = conn.cursor()

curs.execute("""
    SELECT DISTINCT feature_0, count(*) as nb
    FROM train 
    GROUP BY feature_0
""")

for row in curs:
    print(row)
