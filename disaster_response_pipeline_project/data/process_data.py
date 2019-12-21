import sys
import os
import re

import pandas as pd
import numpy as np

import sqlite3
from sqlalchemy import create_engine
#%%
def load_data(messages_filepath, categories_filepath):
    """
    Loads each data set into a pd.DataFrame.

    Params:
    -------
        messages_filepath: file path of messages CSV
        categories_filepath: file path of categories CSV

    Returns:
    --------
        pd.DataFrame of merged data sets, `messages_filepath`
        and `categories_filepath`

    """
    # read data, return pd.DataFrame
    return pd.read_csv(messages_filepath), pd.read_csv(categories_filepath)


def clean_data(datasets):
    """
    Performs operations to clean the input dataframes such that
    the resulting dataframe target array contains 36 categories,
    and one feature column of `messages`.

    Cleaning Steps:
        - Create pd.DataFrame of categories (target array)
            - Extract a list of column names
            - Extract values, 0 or 1
        - Drop original `categories` dataframe
        - Merge `categories` and `messages` dataframes
        - Drop duplicates, axis=0
        - Drop missing values, axis=0

    Params:
    -------
        datasets: DataFrames
            Messages and Category DataFrames

    Returns:
    --------
        pd.DataFrame
            Resulting clean DataFrame

    """
    # unpack datasets
    messages_df, categories_df = datasets

    # merge dataframes
    df = pd.merge(messages_df, categories_df, on='id')

    # create a dataframe of the 36 individual category columns
    categories_df = categories_df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    # use `iloc` to select `row 0` and all columns (`:`)
    row = categories_df.iloc[0, :]

    # use this row to extract a list of new column names for categories
    # replace numbers and hyphen
    category_colnames = row.str.replace("[-0-9]", '').tolist()
    categories_df.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df.loc[:, column].astype('str').apply(lambda x: x[-1])

        # convert column from string to numeric
        categories_df[column] = categories_df[column].astype('int')

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_df], axis=1)

    # drop duplicates
    df.drop_duplicates(subset = 'id', inplace = True)

    # drop column of original messages, 'original'
    df.drop('original', axis=1, inplace=True)

    # amount of missing values is relatively small
    # drop rows with missing values
    df.dropna(subset=['related'], inplace=True)

    # perform a few checks
    # check for missing values
    if df.isnull().sum().any():
        print('WARNING: Contains missing values!')
    # check for duplicate rows
    if df[df.duplicated(subset='id', keep=False)].any().any():
        print('WARNING: Contains duplicate rows!')
    return df



def save_data(df, database_filename):
    """

    Load pd.DataFrame into a SQLite3 database.

    Params:
    -------
        df: DataFrame to load into database
        database_filename: name of database to load

    Returns:
    --------
        NoneType

    """

    # extract directory name
    # dir_ = re.findall(".*/", database_filename)

    # extract table name by stripping away directory name
    # table_name = database_filename.replace('.db', '')

    # Save the clean dataset into an sqlite database.
    conn = sqlite3.connect(f'{database_filename}.db')

    # get a cursor
    cur = conn.cursor()

    # drop table if already exists
    print(f'Lookin for table: {database_filename} ...')
    cur.execute(f"DROP TABLE IF EXISTS {database_filename}")

    # commit and close connection
    conn.commit()
    conn.close()

    # create engine, load to database
    engine = create_engine(f'sqlite:///{database_filename}.db')
    df.to_sql(database_filename, engine, index=False)

    # additionally write a CSV file
    df.to_csv(f"{database_filename}.csv", encoding='utf-8')


def test_database(db_name):
    """
    Test database connection to confirm that it was created
    and the information is accesible.

    Params:
    -------
        db_name: name of database to test

    Returns:
    --------
        Prints a sample of the data contained in specified database.
        dtype: NoneType
    """
    # extract directory name
    # dir_ = re.findall(".*/", db_name)

    # extract table name by stripping away directory name
    # table_name = db_name.replace('.db', '')

    # test connection to database
    conn = sqlite3.connect(f'{db_name}.db')
    cur = conn.cursor()

    try:
        # create the test table including project_id as a primary key
        df = pd.read_sql(f'SELECT * FROM {db_name}', con=conn)
        if any(df):
            print(df.sample(5))
            print('Success! Cleaned data saved to database!')
            print('\n\n')
            conn.commit()
            conn.close()
    except:
        print(f'Database {db_name} not found')
        conn.commit()
        conn.close()


def main():
    """
    Requires Command Line arguments to process data and save
    to a SQLite3 database.

    Command Line arguments:

        messages_filepath: str,
            Filepath to messages DataFrame.

        categories_filepath: str,
            Filepath to categories DataFrame.

        database_filepath: str,
            Filepath for saving to a SQLite database.

    Returns:
    --------
        None.

    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('\n\n')
        print('\nLoading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        files = load_data(messages_filepath, categories_filepath)

        print('\nCleaning data...')
        df = clean_data(files)

        print('\nSaving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print(f'\nTesting database {database_filepath}...\n')
        test_database(database_filepath)


    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()