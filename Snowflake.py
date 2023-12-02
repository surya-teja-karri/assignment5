import snowflake.connector

import json
 
# Read the JSON data from the file
with open('data.json', 'r') as file:
    data = json.load(file)
 
# Snowflake connection parameters
account = ''
user = ''
password = ''
warehouse = 'COMPUTE_WH'
database = 'ASSIGNMENT5'
schema = 'Public'
 
# Establish connection to Snowflake
conn = snowflake.connector.connect(
    user=user,
    password=password,
    account=account,
    warehouse=warehouse,
    database=database,
    schema=schema
)
 
# Create a Snowflake cursor
cur = conn.cursor()
 
# SQL statement to create the table
create_table_sql = '''
CREATE OR REPLACE TABLE your_table_name (
    id VARCHAR,
    "name of image" VARCHAR
)
'''
 
# Execute the SQL statement to create the table
cur.execute(create_table_sql)
 
# Iterate through the JSON data and insert rows into the Snowflake table
for item in data:
    id_val = item['id']
    print(id_val)
    name_of_image = item['name of image']
    print(name_of_image)
   
    # SQL statement to insert data into the table
    insert_sql = '''
    INSERT INTO your_table_name ("ID", "name of image")
    VALUES (%s, %s)
    '''
   
    # Execute the SQL statement to insert data
    cur.execute(insert_sql, (id_val, name_of_image))
 
# Commit the changes
conn.commit()
 
# Close the cursor and connection
cur.close()
conn.close()
