import sqlite3
import psycopg2
import pandas as pd
import numpy as np

# load data from sqlite
sqlite_conn = sqlite3.connect('C:/Liwei/phd/fall 2020/data mining/project/data/FPA_FOD_20170508.sqlite')
from_cursor = sqlite_conn.cursor()
from_cursor.execute("SELECT * FROM 'Fires'")

# connect to psql and upload data
conn = psycopg2.connect(user = "postgres",
                        password = "postgres",
                        host = "localhost",
                        port = "5432",
                        database = "postgres")
to_cursor = conn.cursor()
insert_query = "INSERT INTO fire"
to_cursor.executemany(insert_query, from_cursor.fetchall())