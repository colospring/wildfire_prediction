import sqlite3
import psycopg2
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# load data from sqlite
'''
sqlite_conn = sqlite3.connect('C:/Liwei/data_mining/project/FPA_FOD_20170508.sqlite')
from_cursor = sqlite_conn.cursor()
from_cursor.execute("SELECT * FROM 'Fires'")
'''

# connect to psql and upload data
conn = psycopg2.connect(user="postgres",
                        password="postgres",
                        host="localhost",
                        port="5432",
                        database="postgres")
cursor = conn.cursor()
df = pd.read_sql_query("SELECT * FROM main.analysis", conn)

# load data to python
'''
df = pd.read_sql_query("SELECT * FROM 'Fires'", sqlite_conn)
wf = df[['FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_CODE',
         'STAT_CAUSE_DESCR', 'CONT_DATE', 'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'FIRE_SIZE_CLASS',
         'LATITUDE', 'LONGITUDE', 'STATE', 'COUNTY']].copy()
'''

wf = df.rename(columns=str.lower)
print(wf.dtypes)

# convert date from string to datetime
wf['start_date'] = pd.to_datetime(wf['start_date'], format='%Y-%m-%d')
wf['end_date'] = pd.to_datetime(wf['end_date'], format='%Y-%m-%d')
wf['month'] = pd.DatetimeIndex(df['start_date']).month
wf['dow'] = pd.DatetimeIndex(df['start_date']).dayofweek  # Monday=0, Sunday=6


# check data summary and distribution
pd.set_option('display.max_columns', 10)
wf[['cont_time', 'fire_size', 'bi', 'tmp', 'wind', 'sc', 'erc', 'kbdi', 'pop_dens']].describe()
# distribution plots
wf['fire_year'].plot.hist(title='Fire Year Histogram')
plt.xlabel('Fire Year')
wf['month'].plot.hist(title='Fire Month Histogram')
plt.xlabel('Fire Month')
wf['dow'].plot.hist(title='Fire Day-of-Week Histogram')
plt.xlabel('Day of week')
wf['pop_dens'].plot.hist(title='Population Density Histogram')
plt.xlabel('Population density')
wf['stat_cause_descr'].value_counts().plot(kind='bar', title='Fire Cause Histogram')
plt.xlabel('Fire Cause')
wf['fire_state'].value_counts().plot(kind='bar', title='State Histogram')
plt.xlabel('State')
wf['land_type'].value_counts().plot(kind='bar', title='Land Cover Histogram')
plt.xlabel('Land type')
wf['cont_time'].plot.hist(title='Fire Contained Time Histogram')
plt.xlabel('Fire Contained Time (Day)')
wf.loc[wf['cont_time'] < 50, 'cont_time'].plot.hist(title='Fire Contained Time Histogram')
plt.xlabel('Fire Contained Time (Day)')
wf['fire_size'].plot.hist(title='Fire Size Histogram')
plt.xlabel('Fire Size')
wf[wf['fire_size'] < 100].fire_size.plot.hist(title='Fire Size Histogram')
plt.xlabel('Fire Size')
len(wf[wf['fire_size'] > 100])
wf['fire_size_class'].value_counts().plot(kind='bar', title='Fire Size Class Histogram', rot=0)
plt.xlabel('Fire Size')
sns.kdeplot(wf['bi'], label='BI')
plt.title('Distribution of BI')
sns.kdeplot(wf['tmp'], label='Temperature')
plt.title('Distribution of Temperature')
sns.kdeplot(wf['wind'], label='Wind Speed')
plt.title('Distribution of Wind Speed')

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
outlier_plot(metric='fire_state', xname='State')


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
    plt.figure(figsize=(10, 8))
    # KDE plot of loans that were repaid on time
    sns.kdeplot(df.loc[df[miss].isnull(), metric], label='missing')
    # KDE plot of loans which were not repaid on time
    sns.kdeplot(df.loc[df[miss].notnull(), metric], label = 'not missing')
    # Labeling of plot
    plt.xlabel(metric)
    plt.ylabel(miss)
    plt.title('Distribution')
    plt.legend()


missing_values_plot(wf, miss='end_date', metric='fire_year')
missing_values_plot(wf, miss='end_date', metric='stat_cause_code')
missing_values_plot(wf, miss='end_date', metric='fire_size')
missing_values_plot(wf[wf['fire_size'] < 100], miss='end_date', metric='fire_size')
print('The average fire size with missing contained time is %0.2f' %(wf.loc[wf['cont_time'].isnull(),'fire_size'].mean()))
print('The average fire size without missing contained time is %0.2f' %(wf.loc[wf['cont_time'].notnull(),'fire_size'].mean()))
# plot frequencies of different fire classes by missing value
bins = np.arange(len(wf['fire_size_class'].unique()))
a = wf.loc[wf['cont_time'].isnull(), 'fire_size_class'].value_counts()/len(wf.loc[wf['cont_time'].isnull(), 'fire_size_class'])
b = wf.loc[wf['cont_time'].notnull(), 'fire_size_class'].value_counts()/len(wf.loc[wf['cont_time'].notnull(), 'fire_size_class'])
c = pd.concat([a, b], axis=1).sort_index()
a = c.iloc[:, 0]
b = c.iloc[:, 1]
plt.bar(bins, b, alpha=0.5, label='not missing')
plt.bar(bins, a, alpha=0.5, label='missing')
plt.xticks(bins, a.index.values)
plt.xlabel('fire size class')
plt.title('Distribution')
plt.legend()
plt.show()

# one-hot encoding of categorical variables
# encode land class
len(wf['land_type'].unique())
land_type = pd.get_dummies(wf['land_type'])
wf = pd.concat([wf, land_type.loc[:, land_type.columns != 'barren land']], axis=1, sort=False)
wf = wf.rename(columns=str.lower)
# encode fire cause
len(wf['stat_cause_descr'].unique())
cause = pd.get_dummies(wf['stat_cause_descr'])
wf = pd.concat([wf, cause.loc[:, cause.columns != 'Missing/Undefined']], axis=1, sort=False)
wf = wf.rename(columns=str.lower)
# encode fire size class
len(wf['fire_size_class'].unique())
fire_size = pd.get_dummies(wf['fire_size_class'])
wf = pd.concat([wf, fire_size.loc[:, fire_size.columns != 'G']], axis=1, sort=False)
wf = wf.rename(columns=str.lower)
# encode day of week
len(wf['dow'].unique())
dow = pd.get_dummies(wf['dow'])
dow.columns = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
wf = pd.concat([wf, dow.loc[:, dow.columns != 'sun']], axis=1, sort=False)
wf = wf.rename(columns=str.lower)
# encode state
len(wf['fire_state'].unique())
state = pd.get_dummies(wf['fire_state'])
wf = pd.concat([wf, state.loc[:, state.columns != 'WY']], axis=1, sort=False)

# check data correlation with fire size
wf_nm = wf.drop(columns=['objectid', 'stat_cause_code', 'stat_cause_descr', 'latitude', 'longitude', 'start_date',
                         'end_date', 'fire_geom', 'fire_state', 'dow', 'fire_size_class', 'land_type'])
print(wf_nm.dtypes)
correlations = wf_nm.drop(columns=['a', 'b', 'c', 'd', 'e', 'f']).corr()['fire_size'].sort_values()
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
# check data correlation with contained time
correlations = wf_nm.drop(columns=['a', 'b', 'c', 'd', 'e', 'f']).corr()['cont_time'].sort_values()
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

# check feature correlations
feature_cor = wf_nm.drop(columns=['a', 'b', 'c', 'd', 'e', 'f', 'fire_size', 'cont_time']).corr()
# heatmap of correlation
plt.figure(figsize=(10, 8))
sns.heatmap(feature_cor, cmap=plt.cm.RdYlBu_r, vmin=-0.6, annot=False, vmax=0.6)
plt.title('Correlation Heatmap')  # erc and bi have high correlation, can drop one

# export to csv file
wf_nm.to_csv('C:\Liwei\data_mining\project\cleaned_data.csv',index=False)
