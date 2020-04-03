# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:38:12 2020

@author: smouz

"""
import os
import re
import gc
import time

import sqlite3
from sqlalchemy import create_engine

import numpy as np
import pandas as pd

from joblib import dump, load
# from dill import dump, load

import gensim
from gensim.models import Word2Vec, CoherenceModel

from nltk.data import find
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import ne_chunk, pos_tag


from spellchecker import SpellChecker

from sklearn.impute import KNNImputer, SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

from sklearn.preprocessing import (Normalizer, QuantileTransformer,
                                   PolynomialFeatures, RobustScaler,
                                   StandardScaler, FunctionTransformer)

from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer,
                                             HashingVectorizer
                                             )
from sklearn.feature_selection import SelectKBest, SelectFpr, f_classif, mutual_info_classif


from sklearn.multioutput import MultiOutputClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              ExtraTreesClassifier,
                              BaggingClassifier,
                              RandomTreesEmbedding,
                              StackingClassifier
                              )
from sklearn import svm


from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, classification_report,
                             accuracy_score, make_scorer, log_loss)

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score, KFold)


# this allows to run script from CLI
from custom_transform import SentenceVector, KeywordSearch


def tokenize(text):
    """
    Replace `url` with empty space "".
    Tokenize and lemmatize input `text`.
    Converts to lower case and strips whitespaces.


    Returns:
    --------
        dtype: list, containing processed words
    """

    lemm = WordNetLemmatizer()

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    # load stopwords
    stop_words = stopwords.words("english")

    remove_words = ['one', 'see', 'please', 'thank', 'thank you', 'thanks',
                    'we', 'us', 'you', 'me']
    for addtl_word in remove_words:
        stop_words.append(addtl_word)

    # remove punctuations (retain alphabetical and numeric chars) and convert to all lower case
    # tokenize resulting text
    tokens = word_tokenize(re.sub(r"[^a-zA-Z]", ' ', text.lower().strip()))

    # drop stop words
    no_stops = [word for word in tokens if word not in stop_words]

    # lemmatize and remove stop words
    # lemmatized = [lemm.lemmatize(word) for word in tokens if word not in stop_words]

    return no_stops



def grid_search(model, X_train, y_train, N_JOBS = 6):
    """

    Performs GridSearch to find the best Hyperparameters to maximize
    the F1-score (weighted).

    Params:
    ----------
        model: pipeline object

    Returns:
    -------
        grid_cv: GridSearch object

    """
    print("\nSearching for best model and tuning params... \n")

    rf = RandomForestClassifier()
    ext = ExtraTreesClassifier()

    grid_params = [
        # {
        #     'clf__estimator': [ext],
        #     'clf__estimator__n_estimators': [100, 20],
        #     # 'clf__estimator__max_depth': [None, 4, 8],
        #     # 'clf__estimator__bootstrap': [True, False],
        #     'clf__estimator__n_jobs': [6],
        #     'clf__estimator__random_state': [11]
        #     },

        {
            'clf__estimator': [rf],
            'clf__estimator__n_estimators': [100, 20],
            # 'clf__estimator__max_depth': [None, 4, 8],
            'clf__estimator__class_weight': ['balanced_subsample'],
            'clf__estimator__max_samples': [0.8],
            'clf__estimator__max_features': [0.8],
            'clf__estimator__n_jobs': [6],
            'clf__estimator__random_state': [11]
            }
        ]

    grid_cv = GridSearchCV(
        model,
        grid_params,
        cv=3,
        scoring='recall_weighted',
        n_jobs=N_JOBS,
    )

    grid_cv.fit(X_train, y_train)

    return grid_cv



def load_data(database_filepath, n_sample=5000):
    """
    Import data from database into a DataFrame. Split DataFrame into
    features and predictors, `X` and `Y`. Additionally, extract the names
    of target categories.

    Preprocess data.

    Params:
    -------
        database_filepath: file path of database

    Returns:
    -------
        tuple(X, Y, category_names)
        pd.DataFrame of features and predictors, `X` and `Y`, respectively.
        List of target category names
    """

    engine = create_engine(f'sqlite:///{database_filepath}')

    # extract directory name
    dir_ = re.findall(".*/", database_filepath)

    # extract table name by stripping away directory name
    table_name = database_filepath.replace('.db', '').replace(dir_[0], "")

    df = pd.read_sql_table(f'{table_name}', engine)


    # Sample data
    df = df.sample(n_sample)

    # reset index
    df.reset_index(drop=False, inplace=True)

    # DROP ROWS/COLUMN
    # where sum across entire row is less than 1
    null_idx = np.where(df.loc[:, 'related':].sum(axis=1) < 1)[0]
    # drop rows which contain all null values
    df.drop(null_idx, axis=0, inplace=True)

    # explore `related` feature where its labeled as a `2`
    related_twos = df[df['related'] == 2]
    df.drop(index=related_twos.index, inplace=True)

    # reset index
    df = df.reset_index(drop=True)

    # define features and predictors
    X = df.loc[:, 'message']
    Y = df.loc[:, 'related':]
    Y.drop(Y.nunique()[Y.nunique() < 2].index.tolist(), axis=1, inplace=True)

    # extract label names
    category_names = Y.columns.to_list()

    return X, Y, category_names


def build_model():
    """

    Creates a Pipeline object with preset initial params for estimators
    and classifier.

    Returns:
    -------
        Pipeline object

    """
    N_JOBS = 6

    # LogisticRegression params
    lg_params = dict(
        C = 0.1,
        solver = 'newton-cg',
        penalty = 'l2',
        class_weight = 'balanced',
        multi_class = 'multinomial',
        n_jobs = N_JOBS,
        random_state = 11

    )

    rf_params = dict(
        n_estimators=20,
        # max_depth=4,
        max_features=0.8,
        max_samples=0.8,
        class_weight='balanced_subsample',
        n_jobs=N_JOBS,
        random_state=11
    )

    svc_params = dict(
        C = 0.05,
        kernel = 'rbf',
        gamma = 0.02,
        cache_size = 1000,
        class_weight = 'balanced',
        random_state = 11

        )

    clf = LogisticRegression(**lg_params)
    # clf = RandomForestClassifier(**rf_params)
    # clf = svm.SVC(**svc_params)

    count_vec = CountVectorizer(
    tokenizer=tokenize,
    ngram_range=(1, 1),
    # max_features=1000,
    max_df=0.95,
    min_df=0.01
    )

    sv = SentenceVector(size=100, min_count=2, iter=5, window=1)

    pipeline = Pipeline([
    ('features', FeatureUnion([

            ('count_vec_pipeline', Pipeline([
                    ('count_vect', count_vec),
                    ('tfidf_tx', TfidfTransformer()),
                    ])),
            ('sent_vec_pipeline', Pipeline([
                    ('sentence_vec', sv),
                    ('quant_tx', StandardScaler()),
                    ])),

            ('keywords', KeywordSearch()),

    ], n_jobs=1)),
    # ('norm', Normalizer(norm='l1', copy=False)),
    ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    ])

    return pipeline


def evaluate_model(model, x_test, y_test, category_names):
    """
    Makes predictions on the `x_test` and calculates metrics, `Accuracy`,
    `Precision`, `Recall`, and `F1-score`.

    Inputs `x_test`, and `y_test` are used to compute the scores.

    Results for each label are stored in pd.DataFrame. Scores are
    aggregated and printed to screen.


    Params:
    -------
    model : Pipeline object
        Pipeline to use for making predictions.

    x_test : numpy array
        Predictors test set.

    y_test : numpy array
        Target variables.

    category_names : list
        List of target variable names.

    Returns:
    -------
        NoneType. Simply prints out the scores for each label including
        aggregated scores mean, median and standard deviation.

    """

    y_pred = model.predict(x_test)
    # print label and f1-score for each
    avg = 'weighted'
    f1 = []
    prec = []
    rec = []
    acc = []
    for i in range(y_test[:, :].shape[1]):
        acc.append(accuracy_score(y_test[:, i],y_pred[:, i]))
        f1.append(f1_score(y_test[:, i],y_pred[:, i], average=avg,))
        rec.append(recall_score(y_test[:, i],y_pred[:, i], average=avg))
        prec.append(precision_score(y_test[:, i],y_pred[:, i], average=avg))

    # summarize f1-scores and compare to the rate of positive class occurance
    f1_df = pd.DataFrame({'f1-score': np.round(f1, 4),
                          'precision': np.round(prec, 4),
                          'recall': np.round(rec, 4),
                          'accuracy': np.round(acc, 4)}, index=category_names)

    # print results
    print('\n')
    print('='*75)
    print(f1_df)
    print('\n')
    print('Results Summary:')
    print(f1_df.agg(['mean', 'median', 'std']))
    print('='*75)
    print('\n')



def show_info(X_train, y_train):
    """
    Simply prints the shape of predictors and target arrays.

    Params:
    -------
    X_train : numpy array
        Predictors training subset.

    y_train : numpy array
        Target variables training subset.

    Returns:
    -------
        NoneType. Prints out the shape of predictors and target arrays.

    """
    print("X-shape:", X_train.shape)
    print("Y-shape:", y_train.shape)


def kfold_cv(model, X, y, catg_names, n_splits=3):

    """
    Perform K-Fold cross validation on given feature space and
    target variable. At each split, missing data is filled using
    helper functions to avoid data leakage. Model is fit and predictions
    are made on the training and testing subset.

    Params:
    --------
        X: DataFrame
            Feature space.

        y: Series, DataFrame
            Target variable.

        n_splits: int, default=3
            Number of folds to evaluate.

    Returns:
    ---------
        DataFrame containing metrics on each K-Fold split
    """


    # create splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=11)

    kfold = 0
    for train_idx, test_idx in kf.split(X):
        print('\n')
        print('-'*75)
        print('\nEvaluating Fold:', kfold)

        # define train and test subsets
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        # train and evaluate
        model.fit(X_train.ravel(), y_train)
        evaluate_model(model, X_test.ravel(), y_test.values, catg_names)

        del X_train, X_test, y_train, y_test
        gc.collect()

        kfold += 1

    print('-'*75)



def train_model(X, y, catg_names, params, cv=True, grid_search=False):


    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y.values,
                                                      test_size=0.15)
    _, X_test, _, y_test = train_test_split(X_train,
                                            y_train,
                                            test_size=0.15)

    print("\nValidation set shape:", X_val.shape)
    print("Testing set shape:", X_test.shape)
    print("Training set shape:", X_train.shape)
    gc.collect()

    model = build_model()
    start_time = time.perf_counter()
    print('\nTraining model...')
    print('\nUsing params:\n', model.get_params()['clf__estimator'], '\n')
    model.fit(X_train.ravel(), y_train)
    end_time = time.perf_counter()
    print('\nTraining time:', np.round((end_time - start_time)/60, 4), 'min')

    print('\nEvaluating model...')
    gc.collect()
    evaluate_model(model, X_val.ravel(), y_val, catg_names)
    print('\n---Evaluating Validation set---\n')
    evaluate_model(model, X_test.ravel(), y_test, catg_names)

    # print('\nBest model:', model.best_estimator_)
    if grid_search:
        model = grid_search(model)
    if hasattr(model, 'best_params_'):
        print('\nBest params:', model.best_params_)
        print('\nBest score:', model.best_score_)
        print('Mean scores:', model.cv_results_['mean_test_score'])


    if cv:
        print('\nCross-validating...\n')
        kfold_cv(model, X, y, catg_names, 3)

    # print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
    # save_model(model, model_filepath)
    # print('\nTrained model saved!\n')


def save_model(model, filepath):
    """
    Pickles model to given file path.

    Params:
    -------
        model: Pipeline
            Model to pickle.
        filepath: str
            save model to this directory

    Returns:
    -------
        None.
    """
    try:
        dump(model, filepath)

    except Exception as e:
        print(e)
        print('Failed to pickle model.')


def main(sample_int=5000, gs=False,):
    """
    Command Line arguments:

        database_filepath: str,
            Filepath to database.

        model_filepath: str,
            Filepath to save model.

    Performs the following:
        1. Loads data from provided file path of Database file.
        2. Splits data into training and test subsets.
        3. Trains model and performs GridSearch.
        4. Evaluates model on testing subset.
        5. Saves model to filepath.

    Returns
    -------
        None. Prints results to screen.

    """

    database_filepath = "data/disaster_response.db"
    model_filepath = "models/disaster_clf.pkl"


    print('\nLoading data...\n    DATABASE: {}'.format(database_filepath))

    X, Y, category_names = load_data(database_filepath, n_sample=sample_int)

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      Y.values,
                                                      test_size=0.15)
    _, X_test, _, y_test = train_test_split(X_train,
                                            y_train,
                                            test_size=0.15)

    print("\nValidation set shape:", X_val.shape)
    print("Testing set shape:", X_test.shape)
    print("Training set shape:", X_train.shape)

    # show_info(X_train, y_train)
    del database_filepath
    gc.collect()

    model = build_model()

    if gs:
        model = grid_search(model=model,
                            X_train=X_train,
                            y_train=y_train,
                            N_JOBS = 6)
    if hasattr(model, 'best_params_'):
        print('\nBest params:', model.best_params_)
        print('\nBest score:', model.best_score_)
        print('Mean scores:', model.cv_results_['mean_test_score'])


    else:
        start_time = time.perf_counter()
        print('\nTraining model...')
        print('\nUsing params:\n', model.get_params()['clf__estimator'], '\n')
        model.fit(X_train.ravel(), y_train)
        end_time = time.perf_counter()
        print('\nTraining time:', np.round((end_time - start_time)/60, 4), 'min')

        print('\nEvaluating model...')
        gc.collect()
        evaluate_model(model, X_val.ravel(), y_val, category_names)
        print('\n---Evaluating Validation set---\n')
        evaluate_model(model, X_test.ravel(), y_test, category_names)



    print('\nCross-validating...\n')
    kfold_cv(model, X, Y, category_names, 3)


    print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)
    print('\nTrained model saved!\n')


#%%
if __name__ == '__main__':
    main(sample_int=20000, gs=False)
