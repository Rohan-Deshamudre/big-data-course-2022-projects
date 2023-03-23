import pandas as pd
import numpy as np
import duckdb
from sklearn.linear_model import LinearRegression
from main import *
from training_set_processing import *
from main import con


def add_external_columns(table_name):
    query = '''
        CREATE TABLE merged_''' + table_name + ''' AS 
        SELECT ''' + table_name + '''.*, movielens_data.popularity, movielens_data.vote_average FROM 
        ''' + table_name + ''' LEFT JOIN movielens_data 
        ON ''' + table_name + '''.primaryTitle = movielens_data.title
        AND ''' + table_name + '''.startYear = movielens_data.release_year;
    '''
    return query


def convert_popularity(table_name):
    query = '''
        UPDATE ''' + table_name + ''' SET popularity = 0 WHERE popularity IS NULL;
        ALTER TABLE ''' + table_name + ''' ALTER COLUMN popularity SET DATA TYPE FLOAT;
    '''
    return query


def calculate_missing_popularity(table_name):
    query = '''
        UPDATE ''' + table_name + ''' m1 
        SET popularity = (
          SELECT (SUM(popularity) - MIN(popularity) - MAX(popularity)) / CAST(COUNT(*)-2 as FLOAT) as trimmed_mean 
          FROM ''' + table_name + ''' m2 
          WHERE popularity > 0
        )
        WHERE popularity = 0;
    '''
    return query


def convert_vote_average(table_name):
    query = '''
        UPDATE ''' + table_name + ''' SET vote_average = 0 WHERE vote_average IS NULL;
        ALTER TABLE ''' + table_name + ''' ALTER COLUMN vote_average SET DATA TYPE FLOAT;
    '''
    return query


def predict_missing_vote_average(table_name):
    df_train = con.execute('''
        SELECT * FROM ''' + table_name + '''
        WHERE vote_average != 0
    ''').fetchdf()
    X_train = df_train[['startYear', 'runtimeMinutes', 'numVotes']]
    y_train = df_train['vote_average']

    df_test = con.execute('''
        SELECT * FROM ''' + table_name + '''
        WHERE vote_average = 0
    ''').fetchdf()
    X_test = df_test[['startYear', 'runtimeMinutes', 'numVotes']]

    # Train an SVM classifier
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set and evaluate the performance
    y_pred = model.predict(X_test)
    preds = [max(0, int(a)) for a in y_pred]

    df = con.execute('''
        SELECT * FROM ''' + table_name + '''
    ''').fetchdf()

    df.loc[df['vote_average'] == 0, 'vote_average'] = preds

    con.execute('''
        DROP TABLE IF EXISTS ''' + table_name + '''
    ''')
    con.execute('''
        CREATE TABLE ''' + table_name + ''' AS SELECT * FROM df;
    ''')


def replace_zeros_vote_average(table_name):
    query = '''
    UPDATE ''' + table_name + ''' 
        SET vote_average = (
          SELECT AVG(vote_average) as mean 
          FROM ''' + table_name + ''' 
        )
    WHERE vote_average = 0;
    '''

    return query

