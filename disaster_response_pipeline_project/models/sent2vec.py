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


# =============================================================================

# Custom Transfomer

# =============================================================================
class KeywordSearch(BaseEstimator, TransformerMixin):
    """
    Useses target labels as keywords to generate a sparse matrix
    of keyword occurances within given text.

    """

    def keyword_matrix(self, text):
        """
        Search message for keywords which are obtained from labels.

        Return True if one or more keyword present in text.

        Returns:
        --------
            np.array
        """
        labels = [

                 'related',
                 'request',
                 'offer',
                 'aid',
                 'medical',
                 'medical products',
                 'search',
                 'rescue',
                 'security',
                 'military',
                 'child_alone',
                 'water',
                 'food',
                 'shelter',
                 'clothing',
                 'money',
                 'missing people',
                 'refugee',
                 'death',
                 'infrastructure',
                 'transport',
                 'building',
                 'electricity',
                 'tool',
                 'hospital',
                 'shop',
                 'aid_centers',
                 'weather',
                 'flood',
                 'storm',
                 'fire',
                 'earthquake',
                 'cold',
                 ]
        toks = tokenize(text)

        # check for `labels` keyword in `toks`
        arr = pd.Series(labels).isin(toks).astype(np.int32)

        return arr

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        Transforms input text `X` into a sparse matrix.

        Parameters
        ----------
            X: string
                String or sentence to transform.

        Returns:
        -------
            pd.Series

        """
        return pd.Series(X).apply(self.keyword_matrix)


class SentenceVector(BaseEstimator, TransformerMixin):
    """
    Extracts various named entities from given text input.
    Input can consist of a single string or an array of text
    documents from which to extract the counts of named
    entities.

    """

    def __init__(self, size=100, min_count=5, window=5, iter=5, workers=6):
        self.size = size
        self.min_count = min_count
        self.window = window
        self.workers = workers
        self.window = window
        self.iter = iter

    def _word2vec_model(self, X):
        # tokenize messages, building model
        toks = pd.Series(X).apply(tokenize).tolist()
        wv_model = Word2Vec(toks,
                            size=self.size,
                            min_count=self.min_count,
                            window=self.window,
                            workers=self.workers
                            )

        return wv_model


    def sentence_vector(self, text, model):
        """
        Convert string sentence into a vector representation. Each word is
        converted into a vector and combined with other word vectors.

        If a word is not found in the learned vocabulary, the vector is set
        to zero.

        Params:
        -------
            text : Sentence

        Returns:
        -------
            sentence_vector : numpy array

        """

        text = tokenize(text)
        sentence_vec = np.empty(self.size, dtype=np.float64)

        # iterate over each word in tokenized text
        for word in text:
            # get word vector, if word exists in vocab
            try:
                if sentence_vec.shape[0] < 1:
                    sentence_vec = model.wv[word]
                word_vector = model.wv[word]
                sentence_vec = np.add(word_vector, sentence_vec) / (self.size)

            # if not found in vocab
            except Exception as e:
                # print(e)
                # set word_vector equal to zero
                word_vector = np.zeros_like(sentence_vec.shape)
        return sentence_vec

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        model = self._word2vec_model(X)
        # stack arrays vertically
        stacked = np.vstack([self.sentence_vector(item, model) for item in X])

        # check for missing values
        if np.count_nonzero(stacked) > 0:
            fill_value = np.median(stacked)
            # fill missing values
            stacked = np.where(np.isnan(stacked), fill_value, stacked)

        return stacked


# sv = SentenceVector(size=300, min_count=3, window=1, iter=4, workers=6)

# x_arr3 = sv.transform(X.values[:, 0])
# print(x_arr3.shape)


# print('Missing values:', np.count_nonzero(np.isnan(x_arr3)))

# # fill missing values with zero
# filled = np.where(np.isnan(x_arr3), 0, x_arr3)
# # check NaNs again
# np.count_nonzero(np.isnan(filled))

# # check for infinity values
# np.count_nonzero(np.isinf(x_arr3))


# =============================================================================
#
# =============================================================================


def grid_search(model, grid_params, X_train, y_train, cv=3, N_JOBS=6):
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

    grid_cv = GridSearchCV(
        model,
        grid_params,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=N_JOBS,
    )

    grid_cv.fit(X_train, y_train)

    return grid_cv

def drop_null_idx(df):

    pass

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
        C = 0.2,
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
        # max_features=0.8,
        # max_samples=0.8,
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

    # clf = LogisticRegression(**lg_params)
    clf = RandomForestClassifier(**rf_params)
    # clf = svm.SVC(**svc_params)

    sv = SentenceVector(size=20, min_count=5, window=2)
    count_vec = CountVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        # max_features=1000,
        max_df=0.95,
        min_df=0.01
        )

    # build pipeline
    pipeline = Pipeline([
        ('sent_vec', sv),
        # ('knn_imputer', KNNImputer(n_neighbors=5, weights='distance')),
        # ('imputer', SimpleImputer(strategy='mean')),
        # ('scaler', StandardScaler()),
        ('norm', QuantileTransformer(output_distribution='normal',
                                     random_state=11)),

        ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    ])

    # pipeline = Pipeline([

    # ('features', FeatureUnion([

    #         ('count_vec_pipeline', Pipeline([
    #                 ('count_vect', count_vec),
    #                 # ('tfidf_tx', TfidfTransformer()),
    #                 ])),
    #         ('sent_vec_pipeline', Pipeline([
    #                 ('sentence_vec', sv),
    #                 ('quant_tx', StandardScaler()),
    #                 ('squared', FunctionTransformer(np.square)),
    #                 ])),
    #         # ('keywords', KeywordSearch()),
    #         # ('sentence_vec', sv),

    # ], n_jobs=6)),
    # # ('norm', Normalizer(norm='l1', copy=False)),
    # ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    # ])

    # # return grid search object
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

        n_splits: int, default=4
            Number of folds to evaluate.

    Returns:
    ---------
        DataFrame containing metrics on each K-Fold split
    """


    # create splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=11)
    # models = []

    kfold = 0
    for train_idx, test_idx in kf.split(X):
        print('\n')
        print('-'*75)
        print('\nEvaluating Fold:', kfold)

        # define train data
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]

        # define test data
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]


        # train data
        print('Training model... \n')
        model.fit(X_train.ravel(), y_train)

        print('Evaluating predictions... \n')
        # make predictions
        # y_pred = model.predict(X_test.ravel())
        # y_pred_train = model.predict(X_train.ravel())

        # print metric
        evaluate_model(model, X_test.ravel(), y_test.values, catg_names)

        del X_train, X_test, y_train, y_test
        gc.collect()

        kfold += 1

    print('-'*75)



def train_model(params, cv=True, grid_search=False, sample_int=5000):

    pass



def main(sample_int=5000, grid_search=False,):
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
    N_JOBS = 6
    rf_params = dict(
        n_estimators=6,
        # max_depth=12,
        max_features=0.6,
        max_samples=0.8,
        class_weight='balanced_subsample',
        n_jobs=N_JOBS,
        random_state=11
    )


    database_filepath = "data/disaster_response.db"
    model_filepath = "models/disaster_clf.pkl"


    print('\nLoading data...\n    DATABASE: {}'.format(database_filepath))

    X, Y, category_names = load_data(database_filepath, n_sample=sample_int)

    # sv = SentenceVector(size=300, min_count=3, window=1, iter=4, workers=6)
    # X = sv.transform(X)

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
    del database_filepath, model_filepath
    gc.collect()

    model = build_model()
    # model = stacking_clf(rf_params)

    # start_time = time.perf_counter()
    # print('\nTraining model...')
    # print('\nUsing params:\n', model.get_params()['clf__estimator'], '\n')
    # model.fit(X_train.ravel(), y_train)
    # end_time = time.perf_counter()
    # print('\nTraining time:', np.round((end_time - start_time)/60, 4), 'min')

    # print('\nEvaluating model...')
    # gc.collect()
    # evaluate_model(model, X_val.ravel(), y_val, category_names)
    # print('\n---Evaluating Validation set---\n')
    # evaluate_model(model, X_test.ravel(), y_test, category_names)

    # print('\nBest model:', model.best_estimator_)
    if grid_search:
        model = grid_search(model)
    if hasattr(model, 'best_params_'):
        print('\nBest params:', model.best_params_)
        print('\nBest score:', model.best_score_)
        print('Mean scores:', model.cv_results_['mean_test_score'])

    print('\nCross-validating...\n')
    # gc.collect()
    # cv = KFold(n_splits=3, shuffle=True, random_state=11)
    # scores = cross_val_score(
    #     model,
    #     X.ravel(),
    #     Y.values,
    #     scoring='recall_weighted',
    #     # scoring='recall_macro',
    #     cv=cv,
    #     n_jobs=-1)
    # print('\nCross-val mean score:\n', np.round(np.mean(scores), 4))
    # print("-"*75)

    kfold_cv(model, X, Y, category_names, 3)

    # print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
    # save_model(model, model_filepath)
    # print('\nTrained model saved!\n')


#%%
if __name__ == '__main__':
    main(sample_int=20000)
