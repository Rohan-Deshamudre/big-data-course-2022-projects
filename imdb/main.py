import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, when
from scipy.stats.mstats import winsorize
import duckdb

from imdb.training_set_processing import *
from imdb.external_data_processing import *

con = duckdb.connect(database=':memory:')

con.execute('''
    CREATE TABLE training_set AS SELECT * FROM read_csv_auto('train-1.csv')
''')

con.execute('''
CREATE TABLE movielens_data AS SELECT * FROM read_csv_auto('movielens_data.csv')
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


def execute(query):
    result = con.execute(query).fetchdf()
    return result


def reindex_data(training_set):
    df_1 = con.execute('''SELECT * FROM ''' + training_set + '''''').fetchdf()
    df_1 = df_1.set_index('column0')
    df_2 = con.execute('''SELECT * FROM merged_''' + training_set + '''''').fetchdf()
    df_2 = df_2.set_index('column0')
    df_2 = df_2.reindex(df_1.index)
    return df_2


def run_training():
    execute(replace_missing_startYear('training_set'))
    execute(drop_endYear('training_set'))
    execute(convert_runtimeMins('training_set'))
    execute(calculate_missing_runtimeMins('training_set'))
    execute(convert_numVotes('training_set'))
    predict_missing_numVotes('training_set')
    execute(replace_zeros_numVotes('training_set'))
    execute(drop_originalTitle('training_set'))
    execute(add_external_columns('training_set'))
    execute(convert_popularity('merged_training_set'))
    execute(calculate_missing_popularity('merged_training_set'))
    execute(convert_vote_average('merged_training_set'))
    predict_missing_vote_average('merged_training_set')
    execute(replace_zeros_vote_average('merged_training_set'))
    df = reindex_data('training_set')
    return df



def run_validation():
    execute(replace_missing_startYear('validation_set'))
    execute(drop_endYear('validation_set'))
    execute(convert_runtimeMins('validation_set'))
    execute(calculate_missing_runtimeMins('validation_set'))
    execute(convert_numVotes('validation_set'))
    predict_missing_numVotes('validation_set')
    execute(replace_zeros_numVotes('validation_set'))
    execute(drop_originalTitle('validation_set'))
    execute(add_external_columns('validation_set'))
    execute(convert_popularity('merged_validation_set'))
    execute(calculate_missing_popularity('merged_validation_set'))
    execute(convert_vote_average('merged_validation_set'))
    predict_missing_vote_average('merged_validation_set')
    execute(replace_zeros_vote_average('merged_validation_set'))
    df = reindex_data('validation_set')
    return df


def run_test():
    execute(replace_missing_startYear('test_set'))
    execute(drop_endYear('test_set'))
    execute(convert_runtimeMins('test_set'))
    execute(calculate_missing_runtimeMins('test_set'))
    execute(convert_numVotes('test_set'))
    predict_missing_numVotes('test_set')
    execute(replace_zeros_numVotes('test_set'))
    execute(drop_originalTitle('test_set'))
    execute(add_external_columns('test_set'))
    execute(convert_popularity('merged_test_set'))
    execute(calculate_missing_popularity('merged_test_set'))
    execute(convert_vote_average('merged_test_set'))
    predict_missing_vote_average('merged_test_set')
    execute(replace_zeros_vote_average('merged_test_set'))
    df = reindex_data('test_set')
    return df


def run_models(table_name, model):
    if table_name == 'training_set':
        df_training = run_training()
        if model == 'SVM':
            svm_train_acc, svm_train_pred = train_svm(df_training)
            return svm_train_acc, svm_train_pred
        elif model == 'LogisticRegression':
            lr_train_acc, lr_train_pred = train_logistic_regression(df_training)
            return lr_train_acc, lr_train_pred
        elif model == 'RandomForest':
            rf_train_acc, rf_train_pred = train_random_forest(df_training)
            return rf_train_acc, rf_train_pred
        else:
            gbt_train_acc, gbt_train_pred = gradient_boosted_trees(df_training)
            return gbt_train_acc, gbt_train_pred
    elif table_name == 'validation_set':
        df_validation = run_validation()
        if model == 'SVM':
            svm_val_acc, svm_val_pred = train_svm(df_validation)
            return svm_val_acc, svm_val_pred
        elif model == 'LogisticRegression':
            lr_val_acc, lr_val_pred = train_logistic_regression(df_validation)
            return lr_val_acc, lr_val_pred
        elif model == 'RandomForest':
            rf_val_acc, rf_val_pred = train_random_forest(df_validation)
            return rf_val_acc, rf_val_pred
        else:
            gbt_val_acc, gbt_val_pred = gradient_boosted_trees(df_validation)
            return gbt_val_acc, gbt_val_pred
    else:
        df_test = run_test()
        if model == 'SVM':
            svm_test_acc, svm_test_pred = train_svm(df_test)
            return svm_test_acc, svm_test_pred
        elif model == 'LogisticRegression':
            lr_test_acc, lr_test_pred = train_logistic_regression(df_test)
            return lr_test_acc, lr_test_pred
        elif model == 'RandomForest':
            rf_test_acc, rf_test_pred = train_random_forest(df_test)
            return rf_test_acc, rf_test_pred
        else:
            gbt_test_acc, gbt_test_pred = gradient_boosted_trees(df_test)
            return gbt_test_acc, gbt_test_pred


def main():

    # Run all training_set functions
    train_acc, train_pred = run_models('training_set', 'GradientBoostedTrees')
    print(train_acc)
    pd.DataFrame(train_pred).to_csv('train_pred.csv', index=False, header=False)
    val_acc, val_pred = run_models('validation_set', 'GradientBoostedTrees')
    print(val_acc)
    pd.DataFrame(val_pred).to_csv('val_pred.csv', index=False, header=False)
    test_acc, test_pred = run_models('test_set', 'GradientBoostedTrees')
    print(test_acc)
    pd.DataFrame(test_pred).to_csv('test_pred.csv', index=False, header=False)


if __name__ == "__main__":
    main()