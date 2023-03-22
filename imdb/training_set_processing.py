import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, when
from scipy.stats.mstats import winsorize
import duckdb
from main import *


def execute(query):
    result = con.execute(query).fetchdf()
    return result


def get_table(table_name):
    query = '''
        SELECT * FROM ''' + table_name + '''
    '''
    return query


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


def drop_originalTitle(table_name):
    query = '''
        ALTER TABLE ''' + table_name + '''
        DROP COLUMN originalTitle
    '''
    return query


def drop_column_zero(table_name):
    query = '''
        ALTER TABLE ''' + table_name + '''
        DROP COLUMN column0
    '''
    return query


def convert_numVotes(table_name):
    query = '''
        UPDATE ''' + table_name + ''' SET numVotes = 0 WHERE numVotes IS NULL;
        ALTER TABLE ''' + table_name + ''' ALTER COLUMN numVotes SET DATA TYPE INTEGER;
    '''
    return query


def calculate_missing_numVotes(table_name):
    training_set = get_table(table_name)
    # training_set = training_set.withColumn("numVotes", col("numVotes").cast("float"))
    df = spark.createDataFrame(training_set)

    # Compute the median of the column
    median_value = df.selectExpr(f"percentile_approx({'numVotes'}, 0.5)").collect()[0][0]

    # Replace null values with the median
    df = df.withColumn('numVotes', when(col('numVotes') == 'NaN', median_value).otherwise(col('numVotes')))

    # Compute the winsorized mean of the column
    column = df.select('numVotes').rdd.flatMap(lambda x: x).collect()
    winsorized_mean = winsorize(column, limits=[0.05, 0.05]).mean()

    # Replace null values with winsorized mean
    df = df.withColumn('numVotes', when(col('numVotes') == 'NaN', winsorized_mean).otherwise(col('numVotes')))
    pd_training_set = df.toPandas()

    con.execute('''
        DROP TABLE IF EXISTS training_set
    ''')

    query = '''
        CREATE TABLE ''' + table_name + ''' AS SELECT * FROM ''' + pd_training_set + '''
    '''

    return query


