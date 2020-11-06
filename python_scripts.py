import sqlite3
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# load data from sqlite
sqlite_conn = sqlite3.connect('C:/Liwei/data_mining/project/FPA_FOD_20170508.sqlite')
from_cursor = sqlite_conn.cursor()
from_cursor.execute("SELECT * FROM 'Fires'")

'''
# connect to psql and upload data
conn = psycopg2.connect(user = "postgres",
                        password = "postgres",
                        host = "localhost",
                        port = "5432",
                        database = "postgres")
to_cursor = conn.cursor()
insert_query = "INSERT INTO fire VALUES"
to_cursor.executemany(insert_query, from_cursor.fetchall())
'''

# load data to python
df = pd.read_sql_query("SELECT * FROM 'Fires'", sqlite_conn)
wf = df[['FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_CODE',
         'STAT_CAUSE_DESCR', 'CONT_DATE', 'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'FIRE_SIZE_CLASS',
         'LATITUDE', 'LONGITUDE', 'STATE', 'COUNTY']].copy()
wf = wf.rename(columns=str.lower)
print(wf.dtypes)

# encode fire cause
len(wf['stat_cause_descr'].unique())
# one-hot encoding of categorical variables
cause = pd.get_dummies(wf['stat_cause_descr'])
wf = pd.concat([wf, cause.loc[:, cause.columns != 'Missing/Undefined']], axis=1, sort=False)
wf = wf.rename(columns=str.lower)

# check data summary and distribution
wf[['fire_year', 'discovery_date','fire_size']].describe()
# distribution plots
wf['fire_year'].plot.hist(title = 'Fire Year Histogram')
plt.xlabel('Fire Year')
wf['stat_cause_descr'].value_counts().plot(kind='bar', title = 'Fire Cause Histogram')
plt.xlabel('Fire Cause')
wf['state'].value_counts().plot(kind='bar', title = 'State Histogram')
plt.xlabel('State')
wf['fire_size'].plot.hist(title = 'Fire Size Histogram')
plt.xlabel('Fire Size')
wf[wf['fire_size']<100].fire_size.plot.hist(title = 'Fire Size Histogram')
plt.xlabel('Fire Size')
len(wf[wf['fire_size']>100])

# anomaly analysis
anom = wf[wf['fire_size']>100]
non_anom = wf[wf['fire_size']<100]
# plot fire year distribution
bins = np.linspace(wf['fire_year'].min(), wf['fire_year'].max(), 1)
non_anom['fire_year'].plot.hist(bins, alpha=0.5, weights=np.ones(len(non_anom)) / len(non_anom), label='main')
anom['fire_year'].plot.hist(bins, alpha=0.5, weights=np.ones(len(anom)) / len(anom), label='outliers')
plt.xlabel('Fire Year')
plt.title('Distribution')
plt.legend()
plt.show()


# define a function to plot categorical metrics
def outlier_plot(metric, xname):
    bins = np.arange(len(wf[metric].unique()))
    a = non_anom[metric].value_counts()/len(non_anom)
    b = anom[metric].value_counts()/len(anom)
    c = pd.concat([a, b], axis=1)
    c = c.fillna(0)
    a = c.iloc[:, 0]
    b = c.iloc[:, 1]
    plt.bar(bins, a, alpha=0.5, label='main')
    plt.bar(bins, b, alpha=0.5, label='outliers')
    plt.xticks(bins, a.index.values)
    plt.xticks(rotation=90)
    plt.xlabel(xname)
    plt.title('Distribution')
    plt.legend()
    plt.show()


# plot fire cause distribution
outlier_plot(metric='stat_cause_descr', xname='Fire Cause')
# plot state distribution
outlier_plot(metric='state', xname='State')


# check missing values
# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# Missing values statistics
missing_values = missing_values_table(wf)
print(missing_values)


# check whether missing values are random
def missing_values_plot(df, miss, metric):
    plt.figure(figsize = (10, 8))
    # KDE plot of loans that were repaid on time
    sns.kdeplot(df.loc[df[miss].isnull(), metric], label = 'missing')
    # KDE plot of loans which were not repaid on time
    sns.kdeplot(df.loc[df[miss].notnull(), metric], label = 'not missing')
    # Labeling of plot
    plt.xlabel(metric)
    plt.ylabel(miss)
    plt.title('Distribution')
    plt.legend()
missing_values_plot(wf, miss='cont_date', metric='fire_year')
missing_values_plot(wf, miss='cont_date', metric='discovery_date')
missing_values_plot(wf, miss='cont_date', metric='stat_cause_code')
missing_values_plot(wf, miss='cont_date', metric='fire_size')

# convert time and date from string to datetime
wf['start_date'] = pd.to_datetime(wf['discovery_date'] - pd.Timestamp(0).to_julian_date(), unit='D')
wf['start_time'] = wf['start_date'].astype(str)+' '+wf['discovery_time']
wf['start_time'] = pd.to_datetime(wf['start_time'], format='%Y-%m-%d %H%M')
wf.loc[wf['discovery_time'].isnull(), 'start_time'] = wf['start_date']
wf['end_date'] = pd.to_datetime(wf['cont_date'] - pd.Timestamp(0).to_julian_date(), unit='D')
wf['end_time'] = wf['end_date'].astype(str)+' '+wf['cont_time']
wf['end_time'] = pd.to_datetime(wf['end_time'], format='%Y-%m-%d %H%M')

