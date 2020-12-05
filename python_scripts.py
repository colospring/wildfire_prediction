import sqlite3
import psycopg2
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# modeling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost as xg
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

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
wf['fire_year'].value_counts().sort_index().plot(title='Fire Year Histogram')
plt.xlabel('Fire Year')
wf['month'].value_counts().sort_index().plot(kind='bar', title='Fire Month Histogram')
plt.xlabel('Fire Month')
wf['dow'].value_counts().sort_index().plot(kind='bar', title='Fire Day-of-Week Histogram')
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
# plot fire year distribution
outlier_plot(metric='fire_year', xname='Year')


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
# encode month of year
len(wf['month'].unique())
month = pd.get_dummies(wf['month'])
month.columns = ['jan', 'fed', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
wf = pd.concat([wf, month.loc[:, month.columns != 'dec']], axis=1, sort=False)
wf = wf.rename(columns=str.lower)
# encode fire_year
len(wf['fire_year'].unique())
fire_year = pd.get_dummies(wf['fire_year'])
fire_year.columns = ['y2005', 'y2006', 'y2007', 'y2008', 'y2009', 'y2010', 'y2011', 'y2012', 'y2013', 'y2014', 'y2015']
wf = pd.concat([wf, fire_year.loc[:, fire_year.columns != 'y2005']], axis=1, sort=False)
# encode state
len(wf['fire_state'].unique())
state = pd.get_dummies(wf['fire_state'])
wf = pd.concat([wf, state.loc[:, state.columns != 'WY']], axis=1, sort=False)

# check data correlation with fire size
wf_nm = wf.drop(columns=['objectid', 'stat_cause_code', 'stat_cause_descr', 'latitude', 'longitude', 'start_date',
                         'end_date', 'fire_geom', 'fire_state', 'dow', 'land_type', 'fire_year', 'month'])
print(wf_nm.dtypes)
correlations = wf_nm.drop(columns=['cont_time', 'fire_size_class']).corr()['fire_size'].sort_values()
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
# check data correlation with contained time
correlations = wf_nm.drop(columns=['fire_size', 'fire_size_class']).corr()['cont_time'].sort_values()
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

# check feature correlations
feature_cor = wf_nm.drop(columns=['fire_size_class', 'fire_size', 'cont_time']).corr()
# heatmap of correlation
plt.figure(figsize=(10, 8))
sns.heatmap(feature_cor, cmap=plt.cm.RdYlBu_r, vmin=-0.6, annot=False, vmax=0.6)
plt.title('Correlation Heatmap')  # erc and bi have high correlation, can drop one

# export to csv file
wf_nm.to_csv('C:\Liwei\data_mining\project\cleaned_data.csv',index=False)

# data normalization
wf_norm = wf_nm.copy()
col_norm = ['tmp', 'rh', 'wind', 'erc', 'bi', 'sc', 'kbdi', 'pop_dens']
wf_norm[col_norm] = StandardScaler().fit_transform(wf_norm[col_norm])

## modeling
X = wf_norm.drop(columns=['fire_size_class', 'fire_size', 'cont_time'])
fire_size = np.log(wf_norm['fire_size'])
sns.kdeplot(fire_size, label='Fire size')
fire_class = wf_norm['fire_size_class']
cont_time = wf_norm['cont_time']

# fire size
# build training and test sets
X_train, X_test, fire_size_train, fire_size_test = train_test_split(X, fire_size, test_size=0.3, random_state=42)

# Model 1. Decision tree
# tune the hyperparameters
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 95, num=10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)

# Use the random grid to search for best hyperparameters
# create the base model to tune
tr = DecisionTreeRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
tr_random = RandomizedSearchCV(estimator=tr, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
tr_random.fit(X_train, fire_size_train)
print(tr_random.best_params_)
# {'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 5}

# evaluate the random search
tree = DecisionTreeRegressor()
tree.fit(X_train, fire_size_train)
fire_size_pred = tree.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(fire_size_test, fire_size_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(fire_size_test, fire_size_pred)))
print('R Squared Score is:', r2_score(fire_size_test, fire_size_pred))
'''
Mean Squared Error: 8.218002578076003
Root Mean Squared Error: 2.8667058757528654
R Squared Score is: -0.5513337778685801
'''
# evaluate the best model
best_random = tr_random.best_estimator_
fire_size_pred = best_random.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(fire_size_test, fire_size_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(fire_size_test, fire_size_pred)))
print('R Squared Score is:', r2_score(fire_size_test, fire_size_pred))
'''
Mean Squared Error: 4.845032369719035
Root Mean Squared Error: 2.201143423250524
R Squared Score is: 0.08539060451700486
'''

# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'max_features': ['auto'],
    'min_samples_leaf': [2, 3],
    'min_samples_split': [329, 330, 331]
}

# Create a based model
tr = DecisionTreeRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=tr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# Fit the grid search to the data
grid_search.fit(X_train, fire_size_train)
print(grid_search.best_params_)
# {'max_depth': 9, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 329}

# evaluate the best model
best_random = grid_search.best_estimator_
fire_size_pred = best_random.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(fire_size_test, fire_size_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(fire_size_test, fire_size_pred)))
print('R Squared Score is:', r2_score(fire_size_test, fire_size_pred))

'''
Mean Squared Error: 4.65477184240501
Root Mean Squared Error: 2.1574920260350927
R Squared Score is: 0.12130658042637688
'''
# Extract feature importances
feature_importances = pd.DataFrame(best_random.feature_importances_)
feature_importances.index = X.columns
feature_importances.columns = ['values']
feature_importances = feature_importances.sort_values(by='values', ascending=False)
print('\nFeature Importance:\n', feature_importances.head(15))
# plot test data and prediction
plt.scatter(fire_size_test, fire_size_pred)
plt.plot(fire_size_test, fire_size_test, 'k-')
plt.title('Decision Tree Prediction')
plt.xlabel('Actual')
plt.ylabel('Prediction')

# Model 2. Random forest
# tune hyperparameters
# Number of trees in random forest
n_estimators = [550, 600, 700]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [33, 35, 37]
# Minimum number of samples required to split a node
min_samples_split = [11, 13, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2]
# Method of selecting samples for training each tree
bootstrap = [False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=20, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, fire_size_train)
print(rf_random.best_params_)
# {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
# {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}
# {'n_estimators': 600, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 35, 'bootstrap': False}
# {'n_estimators': 600, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 37, 'bootstrap': False}

# evaluate the random search
rf = RandomForestRegressor()
rf.fit(X_train, fire_size_train)
fire_size_pred = rf.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(fire_size_test, fire_size_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(fire_size_test, fire_size_pred)))
print('R Squared Score is:', r2_score(fire_size_test, fire_size_pred))
'''
Mean Squared Error: 4.273498673573179
Root Mean Squared Error: 2.067244222043728
R Squared Score is: 0.19328051080475928
'''
# evaluate the best model
best_random = rf_random.best_estimator_
fire_size_pred = best_random.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(fire_size_test, fire_size_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(fire_size_test, fire_size_pred)))
print('R Squared Score is:', r2_score(fire_size_test, fire_size_pred))
'''
Mean Squared Error: 4.148330173684553
Root Mean Squared Error: 2.036744994761139
R Squared Score is: 0.21690889494768895
'''

# Extract feature importances
feature_importances = pd.DataFrame(best_random.feature_importances_)
feature_importances.index = X.columns
feature_importances.columns = ['values']
feature_importances = feature_importances.sort_values(by='values', ascending=False)
print('\nFeature Importance:\n', feature_importances.head(15))
# plot test data and prediction
plt.scatter(fire_size_test, fire_size_pred)
plt.plot(fire_size_test, fire_size_test, 'k-')
plt.title('Random Forest Prediction')
plt.xlabel('Actual')
plt.ylabel('Prediction')

# Model 3. XGBoost
# tune hyperparameters
parameters = {'nthread': [4], #when use hyperthread, xgboost may become slower
              'learning_rate': [.04, .03, .02], #so called `eta` value
              'max_depth': [11],
              'min_child_weight': [2],
              'subsample': [0.9, 1],
              'colsample_bytree': [0.9, 1],
              'n_estimators': [550]}
print(parameters)
xgb = xg.XGBRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across different combinations, and use all available cores
xgb_random = RandomizedSearchCV(xgb, param_distributions=parameters, cv=3, n_iter=10, verbose=2, n_jobs=-1)
# Fit the random search model
xgb_random.fit(X_train, fire_size_train)
print(xgb_random.best_params_)
# {'subsample': 0.9, 'nthread': 4, 'n_estimators': 500, 'min_child_weight': 2, 'max_depth': 7, 'learning_rate': 0.07, 'colsample_bytree': 0.9}
# {'subsample': 0.9, 'nthread': 4, 'n_estimators': 600, 'min_child_weight': 2, 'max_depth': 9, 'learning_rate': 0.06, 'colsample_bytree': 0.9}
# {'subsample': 0.9, 'nthread': 4, 'n_estimators': 550, 'min_child_weight': 2, 'max_depth': 11, 'learning_rate': 0.04, 'colsample_bytree': 0.9}
# {'subsample': 0.9, 'nthread': 4, 'n_estimators': 550, 'min_child_weight': 2, 'max_depth': 11, 'learning_rate': 0.03, 'colsample_bytree': 0.9}

# evaluate the random search
xgb = xg.XGBRegressor()
xgb.fit(X_train, fire_size_train)
fire_size_pred = xgb.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(fire_size_test, fire_size_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(fire_size_test, fire_size_pred)))
print('R Squared Score is:', r2_score(fire_size_test, fire_size_pred))
'''
Mean Squared Error: 4.236333972051183
Root Mean Squared Error: 2.058235645413611
R Squared Score is: 0.200296188430523
'''
# evaluate the best model
best_random = xgb_random.best_estimator_
fire_size_pred = best_random.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(fire_size_test, fire_size_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(fire_size_test, fire_size_pred)))
print('R Squared Score is:', r2_score(fire_size_test, fire_size_pred))
'''
Mean Squared Error: 4.1144816937648665
Root Mean Squared Error: 2.0284185203662646
R Squared Score is: 0.22329856077342158
'''

# Extract feature importances
feature_importances = pd.DataFrame(best_random.feature_importances_)
feature_importances.index = X.columns
feature_importances.columns = ['values']
feature_importances = feature_importances.sort_values(by='values', ascending=False)
print('\nFeature Importance:\n', feature_importances.head(15))
# plot test data and prediction
plt.scatter(fire_size_test, fire_size_pred)
plt.plot(fire_size_test, fire_size_test, 'k-')
plt.title('XGBoost Prediction')
plt.xlabel('Actual')
plt.ylabel('Prediction')

# fire size class
# build training and test sets
X_train, X_test, fire_class_train, fire_class_test = train_test_split(X, fire_class, test_size=0.3,
                                                                      stratify=fire_class, random_state=42)

# check fire size by class
ax = sns.boxplot(x="fire_size_class", y="fire_size",
                 data=wf_nm.loc[(wf_nm['fire_size_class']!='G') & (wf_nm['fire_size_class']!='F'),], showfliers=False,
                 order=['A','B','C','D','E'])
ax.set_title('Fire Size by Class')
ax.set_ylabel('fire size')
ax.set_xlabel('class')

# Model 1. Decision tree
# tune the hyperparameters
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 95, num=10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)

# Use the random grid to search for best hyperparameters
# create the base model to tune
tr = DecisionTreeClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
tr_random = RandomizedSearchCV(estimator=tr, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
tr_random.fit(X_train, fire_class_train)
print(tr_random.best_params_)
# {'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 15}

# evaluate the random search
tree = DecisionTreeClassifier(max_depth=15, max_features='auto', min_samples_leaf=4, min_samples_split=2)
tree.fit(X_train, fire_class_train)
fire_class_pred = tree.predict(X_test)
print('Accuracy Score:', accuracy_score(fire_class_test, fire_class_pred))
print('Recall Score:', recall_score(fire_class_test, fire_class_pred, average='weighted'))
print('F1 Score:', f1_score(fire_class_test, fire_class_pred, average='weighted'))
'''
Accuracy Score: 0.5294502891381593
Recall Score: 0.5294502891381593
F1 Score: 0.5290431325686366
'''
# evaluate the best model
best_random = tr_random.best_estimator_
fire_class_pred = best_random.predict(X_test)
print('Accuracy Score:', accuracy_score(fire_class_test, fire_class_pred))
print('Recall Score:', recall_score(fire_class_test, fire_class_pred, average='weighted'))
print('F1 Score:', f1_score(fire_class_test, fire_class_pred, average='weighted'))
'''
Accuracy Score: 0.595067233331011
Recall Score: 0.595067233331011
F1 Score: 0.5589545333865262
'''

# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [10, 15, 20],
    'max_features': ['auto'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 3]
}

# Create a based model
tr = DecisionTreeClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=tr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# Fit the grid search to the data
grid_search.fit(X_train, fire_class_train)
print(grid_search.best_params_)
# {'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 2}

# evaluate the best model
best_random = grid_search.best_estimator_
fire_class_pred = best_random.predict(X_test)
print('Accuracy Score:', accuracy_score(fire_class_test, fire_class_pred))
print('Recall Score:', recall_score(fire_class_test, fire_class_pred, average='weighted'))
print('F1 Score:', f1_score(fire_class_test, fire_class_pred, average='weighted'))

'''
Accuracy Score: 0.5946073991500035
Recall Score: 0.5946073991500035
F1 Score: 0.559657711217595
'''
# Extract feature importances
feature_importances = pd.DataFrame(best_random.feature_importances_)
feature_importances.index = X.columns
feature_importances.columns = ['values']
feature_importances = feature_importances.sort_values(by='values', ascending=False)
print('\nFeature Importance:\n', feature_importances.head(15))
# compute ROC-AUC score
y_pred = pd.get_dummies(pd.DataFrame(fire_class_pred))
y_test = pd.get_dummies(pd.DataFrame(fire_class_test))
print('ROC-AUC Score:', roc_auc_score(y_test, y_pred, multi_class='ovo', average='macro')) # 0.5360179495279082
# Compute ROC curve and ROC area for each class
fpr_tr = dict()
tpr_tr = dict()
roc_auc_tr = dict()
for i in range(len(y_test. columns)):
    fpr_tr[i], tpr_tr[i], _ = roc_curve(y_test.iloc[:, i], y_pred.iloc[:, i])
    roc_auc_tr[i] = auc(fpr_tr[i], tpr_tr[i])

# Model 2. Random forest
# tune hyperparameters
# Number of trees in random forest
n_estimators = [500]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [33, 35, 37]
# Minimum number of samples required to split a node
min_samples_split = [10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
rf_random.fit(X_train, fire_class_train)
print(rf_random.best_params_)
# {'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': True}
# {'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 35, 'bootstrap': True}

# evaluate the random search
rf = RandomForestClassifier()
rf.fit(X_train, fire_class_train)
fire_class_pred = rf.predict(X_test)
print('Accuracy Score:', accuracy_score(fire_class_test, fire_class_pred))
print('Recall Score:', recall_score(fire_class_test, fire_class_pred, average='weighted'))
print('F1 Score:', f1_score(fire_class_test, fire_class_pred, average='weighted'))
'''
Accuracy Score: 0.620553194454121
Recall Score: 0.620553194454121
F1 Score: 0.5918967415512428
'''
# evaluate the best model
best_random = rf_random.best_estimator_
fire_class_pred = best_random.predict(X_test)
print('Accuracy Score:', accuracy_score(fire_class_test, fire_class_pred))
print('Recall Score:', recall_score(fire_class_test, fire_class_pred, average='weighted'))
print('F1 Score:', f1_score(fire_class_test, fire_class_pred, average='weighted'))
'''
Accuracy Score: 0.6343064167769804
Recall Score: 0.6343064167769804
F1 Score: 0.5966317199898747
'''

# Extract feature importances
feature_importances = pd.DataFrame(best_random.feature_importances_)
feature_importances.index = X.columns
feature_importances.columns = ['values']
feature_importances = feature_importances.sort_values(by='values', ascending=False)
print('\nFeature Importance:\n', feature_importances.head(15))

# compute ROC-AUC score
y_pred = pd.get_dummies(pd.DataFrame(fire_class_pred))
y_test = pd.get_dummies(pd.DataFrame(fire_class_test))
print('ROC-AUC Score:', roc_auc_score(y_test, y_pred, multi_class='ovo', average='macro')) # 0.5467949571444368
# Compute ROC curve and ROC area for each class
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i in range(len(y_test.columns)):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test.iloc[:, i], y_pred.iloc[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# Model 3. XGBoost
# tune hyperparameters
parameters = {'nthread': [4], #when use hyperthread, xgboost may become slower
              'learning_rate': [.04, .03, .02], #so called `eta` value
              'max_depth': [9, 11, 13],
              'min_child_weight': [2, 5],
              'subsample': [0.9, 0.7],
              'colsample_bytree': [0.9, 0.7],
              'n_estimators': [500, 400]}
print(parameters)
xgb = xg.XGBClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across different combinations, and use all available cores
xgb_random = RandomizedSearchCV(xgb, param_distributions=parameters, cv=3, n_iter=30, verbose=2, n_jobs=-1)
# Fit the random search model
xgb_random.fit(X_train, fire_class_train)
print(xgb_random.best_params_)
# {'subsample': 0.9, 'nthread': 4, 'n_estimators': 500, 'min_child_weight': 2, 'max_depth': 11, 'learning_rate': 0.03, 'colsample_bytree': 0.9}

# evaluate the random search
xgb = xg.XGBClassifier(subsample=0.9, nthread=4, n_estimators=500, min_child_weight=2, max_depth=11, learning_rate=0.03, colsample_bytree=0.9)
xgb.fit(X_train, fire_class_train)
fire_class_pred = xgb.predict(X_test)
print('Accuracy Score:', accuracy_score(fire_class_test, fire_class_pred))
print('Recall Score:', recall_score(fire_class_test, fire_class_pred, average='weighted'))
print('F1 Score:', f1_score(fire_class_test, fire_class_pred, average='weighted'))
'''
Accuracy Score: 0.6299588936110918
Recall Score: 0.6299588936110918
F1 Score: 0.5934226758915057
'''
# evaluate the best model
best_random = xgb_random.best_estimator_
fire_class_pred = best_random.predict(X_test)
print('Accuracy Score:', accuracy_score(fire_class_test, fire_class_pred))
print('Recall Score:', recall_score(fire_class_test, fire_class_pred, average='weighted'))
print('F1 Score:', f1_score(fire_class_test, fire_class_pred, average='weighted'))
'''
Accuracy Score: 0.6364105065143175
Recall Score: 0.6364105065143175
F1 Score: 0.6008914532709038
'''

# Extract feature importances
feature_importances = pd.DataFrame(best_random.feature_importances_)
feature_importances.index = X.columns
feature_importances.columns = ['values']
feature_importances = feature_importances.sort_values(by='values', ascending=False)
print('\nFeature Importance:\n', feature_importances.head(15))

# compute ROC-AUC score
y_pred = pd.get_dummies(pd.DataFrame(fire_class_pred))
y_test = pd.get_dummies(pd.DataFrame(fire_class_test))
print('ROC-AUC Score:', roc_auc_score(y_test, y_pred, multi_class='ovo', average='macro')) # 0.5500003570578348
# Compute ROC curve and ROC area for each class
fpr_xgb = dict()
tpr_xgb = dict()
roc_auc_xgb = dict()
for i in range(len(y_test.columns)):
    fpr_xgb[i], tpr_xgb[i], _ = roc_curve(y_test.iloc[:, i], y_pred.iloc[:, i])
    roc_auc_xgb[i] = auc(fpr_xgb[i], tpr_xgb[i])

# plot ROC curves
for i in range(len(y_test.columns)):
    plt.figure()
    plt.plot(fpr_tr[i], tpr_tr[i], 'b', label='Decision Tree ROC curve (area = %0.2f)' % roc_auc_tr[i])
    plt.plot(fpr_rf[i], tpr_rf[i], 'g', label='Random Forest ROC curve (area = %0.2f)' % roc_auc_rf[i])
    plt.plot(fpr_xgb[i], tpr_xgb[i], 'y', label='XGBoost ROC curve (area = %0.2f)' % roc_auc_xgb[i])
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")
    plt.show()

# improvements by reducing fire classes
# summary statistics of fire size by fire classes
pd.set_option('display.max_columns', 10)
wf_nm.groupby(['fire_size_class'])['fire_size'].describe()
wf_nm['fire_size'].describe()
# combine classes C-G as one class
fire_class_new = pd.DataFrame(fire_class.copy())
fire_class_new['class'] = 'large'
fire_class_new.loc[fire_class_new['fire_size_class'] == 'A', 'class'] = 'small'
# fire_class_new.loc[fire_class_new['fire_size_class'] == 'B', 'class'] = 'medium'
fire_class_new = fire_class_new.drop(columns=['fire_size_class'])
X_train, X_test, fire_class_train, fire_class_test = train_test_split(X, fire_class_new, test_size=0.3,
                                                                      stratify=fire_class_new, random_state=42)
# fit XGBoost model
xgb = xg.XGBClassifier(subsample=0.9, nthread=4, n_estimators=500, min_child_weight=2, max_depth=11,
                       learning_rate=0.03, colsample_bytree=0.9)
model = xgb.fit(X_train, fire_class_train.values.ravel())
fire_class_pred = model.predict(X_test)
print('Accuracy Score:', accuracy_score(fire_class_test, fire_class_pred))
# print('Recall Score:', recall_score(fire_class_test, fire_class_pred, average='weighted'))
print('Recall Score:', recall_score(fire_class_test, fire_class_pred, pos_label="large"))
# print('F1 Score:', f1_score(fire_class_test, fire_class_pred, average='weighted'))
print('F1 Score:', f1_score(fire_class_test, fire_class_pred, pos_label="large"))
'''
3 classes:
Accuracy Score: 0.6394342646136696
Recall Score: 0.6394342646136696
F1 Score: 0.6160582844688224
'''
'''
2 classes:
Accuracy Score: 0.6993520518358531
Recall Score: 0.6488551924122515
F1 Score: 0.6656853326722241
'''
# Extract feature importances
feature_importances = pd.DataFrame(model.feature_importances_)
feature_importances.index = X.columns
feature_importances.columns = ['values']
feature_importances = feature_importances.sort_values(by='values', ascending=False)
print('\nFeature Importance:\n', feature_importances.head(15))

# compute ROC-AUC score
y_pred = pd.get_dummies(pd.DataFrame(fire_class_pred))
y_test = pd.get_dummies(pd.DataFrame(fire_class_test))
print('ROC-AUC Score:', roc_auc_score(y_test, y_pred, multi_class='ovo', average='macro'))
'''
3 classes: 0.6289476118209149
2 classes: 0.695725353520076
'''

# Compute ROC curve and ROC area for each class
fpr_xgb2 = dict()
tpr_xgb2 = dict()
roc_auc_xgb2 = dict()
for i in range(len(y_test.columns)):
    fpr_xgb2[i], tpr_xgb2[i], _ = roc_curve(y_test.iloc[:, i], y_pred.iloc[:, i])
    roc_auc_xgb2[i] = auc(fpr_xgb2[i], tpr_xgb2[i])

# focus only on California
wf_ca = wf_norm.loc[wf_norm['CA'] == 1, ]
wf_ca = wf_ca.drop(columns=['AZ', 'CA', 'CO', 'FL', 'GA', 'ID', 'MT', 'NC', 'NM', 'NV', 'OR', 'SC', 'UT', 'WA'])
wf_ca['class'] = 'large'
wf_ca.loc[wf_ca['fire_size_class'] == 'A', 'class'] = 'small'
wf_ca = wf_ca.drop(columns=['fire_size_class'])
# split data to training and test sets
X_ca = wf_ca.drop(columns=['class', 'fire_size', 'cont_time'])
y_ca = wf_ca['class']
X_train, X_test, y_train, y_test = train_test_split(X_ca, y_ca, test_size=0.3, stratify=y_ca, random_state=42)
# model fitting
xgb = xg.XGBClassifier(subsample=0.9, nthread=4, n_estimators=500, min_child_weight=2, max_depth=11,
                       learning_rate=0.03, colsample_bytree=0.9)
model = xgb.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, y_pred))
print('Recall Score:', recall_score(y_test, y_pred, pos_label="large"))
print('F1 Score:', f1_score(y_test, y_pred, pos_label="large"))
'''
Accuracy Score: 0.6549288961981232
Recall Score: 0.6433475967757986
F1 Score: 0.6444377990430623
'''
# Extract feature importances
feature_importances = pd.DataFrame(model.feature_importances_)
feature_importances.index = X_ca.columns
feature_importances.columns = ['values']
feature_importances = feature_importances.sort_values(by='values', ascending=False)
print('\nFeature Importance:\n', feature_importances.head(15))

# compute ROC-AUC score
y_pred = pd.get_dummies(pd.DataFrame(fire_class_pred))
y_test = pd.get_dummies(pd.DataFrame(fire_class_test))
print('ROC-AUC Score:', roc_auc_score(y_test, y_pred, multi_class='ovo', average='macro')) # 0.695725353520076
# Compute ROC curve and ROC area for each class
fpr_xgb2 = dict()
tpr_xgb2 = dict()
roc_auc_xgb2 = dict()
for i in range(len(y_test.columns)):
    fpr_xgb2[i], tpr_xgb2[i], _ = roc_curve(y_test.iloc[:, i], y_pred.iloc[:, i])
    roc_auc_xgb2[i] = auc(fpr_xgb2[i], tpr_xgb2[i])