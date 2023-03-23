import pandas as pd
import numpy as np
import duckdb
from sklearn.linear_model import LinearRegression
from imdb.main import *
from main import con





# def create_training_set_table(table_name, file_name):
#     query = '''
#         CREATE TABLE ''' + table_name + ''' AS SELECT * FROM read_csv_auto(''' + file_name + ''')
#     '''
#     return query

#
# def add_data(table_name, file_name):
#     query = '''
#         INSERT INTO ''' + table_name + ''' SELECT * FROM read_csv_auto(''' + file_name + ''')
#     '''
#     return query


def replace_missing_startYear(table_name):
    query = '''
        UPDATE ''' + table_name + '''
        SET startYear = endYear
        WHERE startYear = '\\N'
    '''
    return query


def drop_endYear(table_name):
    query = '''
        ALTER TABLE ''' + table_name + '''
        DROP COLUMN endYear
    '''
    return query


def convert_runtimeMins(table_name):
    query = '''
        UPDATE ''' + table_name + ''' SET runtimeMinutes = 0 WHERE runtimeMinutes = '\\N';
        ALTER TABLE ''' + table_name + ''' ALTER COLUMN runtimeMinutes SET DATA TYPE INTEGER;
    '''
    return query


def calculate_missing_runtimeMins(table_name):
    query = '''
        UPDATE ''' + table_name + ''' m1 
        SET runtimeMinutes = (
          SELECT AVG(runtimeMinutes) as yearly_mean 
          FROM ''' + table_name + ''' m2 
          WHERE m1.startYear = m2.startYear AND runtimeMinutes > 0 
          GROUP BY m2.startYear
        )
        WHERE runtimeMinutes = 0;
    '''
    return query


def convert_numVotes(table_name):
    query = '''
        UPDATE ''' + table_name + ''' SET numVotes = 0 WHERE numVotes IS NULL;
        ALTER TABLE ''' + table_name + ''' ALTER COLUMN numVotes SET DATA TYPE INTEGER;
    '''
    return query


def predict_missing_numVotes(input_name):
    df_train = con.execute('''
        SELECT * FROM ''' + input_name + '''
        WHERE numVotes != 0
    ''').fetchdf()
    X_train = df_train[['startYear', 'runtimeMinutes']]
    y_train = df_train['numVotes']

    df_test = con.execute('''
        SELECT * FROM ''' + input_name + '''
        WHERE numVotes = 0
    ''').fetchdf()
    X_test = df_test[['startYear', 'runtimeMinutes']]

    # Train an SVM classifier
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set and evaluate the performance
    y_pred = model.predict(X_test)
    preds = [max(0, int(a)) for a in y_pred]

    df = con.execute('''
        SELECT * FROM ''' + input_name + '''
    ''').fetchdf()

    df.loc[df['numVotes'] == 0, 'numVotes'] = preds

    con.execute('''
        DROP TABLE IF EXISTS ''' + input_name + '''
    ''')
    con.execute('''
        CREATE TABLE ''' + input_name + ''' AS SELECT * FROM df;
    ''')


def replace_zeros_numVotes(input_name):
    query = '''
    UPDATE ''' + input_name + ''' 
        SET numVotes = (
          SELECT AVG(numVotes) as mean 
          FROM ''' + input_name + ''' 
        )
    WHERE numVotes = 0;
    '''

    return query


def drop_originalTitle(table_name):
    query = '''
        ALTER TABLE ''' + table_name + '''
        DROP COLUMN originalTitle
    '''
    return query

