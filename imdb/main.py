import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, when
from scipy.stats.mstats import winsorize
import duckdb

from imdb.training_set_processing import *

# Setting up duckDB connectiojn

con = duckdb.connect(database=':memory:')

con.execute('''
    CREATE TABLE training_set AS SELECT * FROM read_csv_auto('train-1.csv')
''')

con.execute('''
    INSERT INTO training_set SELECT * FROM read_csv_auto('train-2.csv');
    INSERT INTO training_set SELECT * FROM read_csv_auto('train-3.csv');
    INSERT INTO training_set SELECT * FROM read_csv_auto('train-4.csv');
    INSERT INTO training_set SELECT * FROM read_csv_auto('train-5.csv');
    INSERT INTO training_set SELECT * FROM read_csv_auto('train-6.csv');
    INSERT INTO training_set SELECT * FROM read_csv_auto('train-7.csv');
    INSERT INTO training_set SELECT * FROM read_csv_auto('train-8.csv')
''')

# Setting up Spark Session
spark = SparkSession \
    .builder \
    .appName("Big Data Project") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


def main():
    # execute(create_training_set_table('training_set', 'train-1.csv'))
    # execute(add_data('train-2.csv'))
    # execute(add_data('train-3.csv'))
    # execute(add_data('train-4.csv'))
    # execute(add_data('train-5.csv'))
    # execute(add_data('train-6.csv'))
    # execute(add_data('train-7.csv'))
    # execute(add_data('train-8.csv'))

    execute(replace_missing_startYear('training_set'))
    execute(drop_endYear('training_set'))
    execute(convert_runtimeMins('training_set'))
    execute(calculate_missing_runtimeMins('training_set'))
    execute(drop_originalTitle('training_set'))
    execute(drop_column_zero('training_set'))
    execute(convert_numVotes('training_set'))
    execute(calculate_missing_numVotes('training_set'))
    execute(gradient_boosted_trees('training_set'))


if __name__ == "__main__":
    main()