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
                                   PolynomialFeatures, RobustScaler, StandardScaler)
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer,
                                             HashingVectorizer
                                             )


from sklearn.multioutput import MultiOutputClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              ExtraTreesClassifier,
                              BaggingClassifier,
                              RandomTreesEmbedding,
                              StackingClassifier
                              )


from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, classification_report,
                             accuracy_score, make_scorer, log_loss)

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


pd.options.display.max_columns = 60
pd.options.display.max_rows = 60


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


#%%
# load data from database
engine = create_engine('sqlite:///data/disaster_response.db')
df = pd.read_sql_table('disaster_response', engine)


X = df.iloc[:, 1]
# 'related' and everything after it
Y = df.iloc[:, 3:]

# X, Y, category_names = load_data('sqlite:///../data/disaster_response.db')


#%%

# =============================================================================

# Explore and clean messages

# =============================================================================

df.head()
df.shape

df = df.sample(25000)
df.reset_index(drop=False, inplace=True)

### DROP ROWS
# where sum across entire row is less than 1
null_idx = np.where(df.loc[:, 'related':].sum(axis=1) < 1)[0]

# drop rows which contain all null values
df.drop(null_idx, axis=0, inplace=True)

# drop rows where 'related' has a numeric 2
related_twos = df[df['related'] == 2]
df.drop(index=related_twos.index, inplace=True)


df.shape



### TEST

# ISSUE:
#   - KeyError; index not found in axis
# SOLUTION:
#   - reset index after sampling the DataFrame, this way `np.where` will return
#   the appropriate index

# NOTE:
#   - np.where returns relatve indecies of positive occurance. It does not
#   use the DataFrame indices.
#   - therefore, DataFrame indices must be reset



df = df.sample(5000)
df.reset_index(drop=False, inplace=True)
dfs = df.iloc[:10, :]
nidx = np.where(dfs.loc[:, 'related':].sum(axis=1) < 1)[0]
dfs.loc[nidx, 'message':]


#%%
# =============================================================================

# SpellChecker

# =============================================================================


# spell = SpellChecker()
# spell.correction('comitee')
# spell.candidates('comitee')

# spell.correction('destroed')
# spell.candidates('destroed')


#%%

# =============================================================================
#
# Word 2 Vec

# =============================================================================
# sample model
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)


model.word_vec('fire').shape
model.word_vec('fire').reshape(1, -1).shape

model.most_similar(['tent', 'water'], ['shelter'], topn=5)

msg = '''A Comitee in Delmas 19, Rue ( street ) Janvier, Impasse Charite #2.
We have about 500 people in a temporary shelter and we are in dire need
of Water, Food, Medications, Tents and Clothes. Please stop by and see us.'''

# msg = df.loc[13, 'message']
print(msg)
tokenize(msg)

model.most_similar(['shelter'], topn=5)

#%%

# =============================================================================

# Train New Model

# =============================================================================
X.iloc[:5, 0].apply(tokenize).tolist()

toks = X.iloc[:, 0].apply(tokenize).tolist()

# sentences = [sent.split(',') for sent in items]

model = Word2Vec(toks, size=100,  min_count=2)

# most similar words
model.wv.most_similar(['water', 'shelter'], topn=5)
model.wv['water']


model.wv.most_similar(['thomassin'], topn=5)

### VECTORIZE A SENTENCE

def sentence_vector(text):
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
    sentence_vec = np.empty(100)

    # iterate over each word in tokenized text
    for word in text:
        # get word vector, if word exists in vocab
        try:
            if sentence_vec.shape[0] < 1:
                sentence_vec = model.wv[word]
            word_vector = model.wv[word]
            sentence_vec = np.add(word_vector, sentence_vec) / 100

        # if not found in vocab
        except Exception as e:
            # print(e)
            # set word_vector equal to zero
            word_vector = np.zeros_like(sentence_vec.shape)
    return sentence_vec

#%%

sentence_vector(msg)
tokenize(msg)
model.wv['let'][:5]

X.iloc[:15, 0].apply(sentence_vector).apply(np.shape)
np.vstack(X.iloc[:15, 0].apply(sentence_vector))

X.iloc[:15, 0].apply(sentence_vector)[4]


# Stack vertically, all vectors
# only works if all vectors are the same shape
x_arr = np.vstack([sentence_vector(item) for item in X.values[:, 0]])
x_arr.shape


# Word not found
sentence_vector(X.values[66, 0])
model.wv['insecure']

#%%

# =============================================================================

# Custom Transfomer

# =============================================================================
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

    remove_words = ['one', 'see', 'please', 'thank', 'thank you', 'thanks']
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
                sentence_vec = np.add(word_vector, sentence_vec) / self.size

            # if not found in vocab
            except Exception as e:
                # print(e)
                # set word_vector equal to zero
                word_vector = np.zeros_like(sentence_vec.shape, dtype=np.float64)
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



#%%

# =============================================================================

# Compare timing for two methods

# =============================================================================

# import time

# # Stack vertically, all vectors
# # only works if all vectors are the same shape
# print('X shape:', X.shape)
# start_time = time.perf_counter()
# x_arr = np.vstack([sentence_vector(item) for item in X.values[:, 0]])
# end_time = time.perf_counter()
# print("\nList comprehension time:", np.round(end_time - start_time, 4))


# start_time = time.perf_counter()
# x_arr2 = np.vstack(X.iloc[:, 0].apply(sentence_vector))
# end_time = time.perf_counter()
# print("\nApply method time:", np.round(end_time - start_time, 4))


#%%


# =============================================================================

# Prepare Data for Training

# =============================================================================
# Vectorize each sentence in feature space, X
# x_arr = np.vstack([sentence_vector(item) for item in X.values[:, 0]])





#%%

# =============================================================================

# Scit-kit Learn Pipeline

# =============================================================================


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

    # DROP ROWS/COLUMN
    # explore `related` feature where its labeled as a `2`
    related_twos = df[df['related'] == 2]
    df.drop(index=related_twos.index, inplace=True)

    # # where sum across entire row is less than 1
    # null_idx = np.where(df.loc[:, 'related':].sum(axis=1) < 1)[0]
    # # drop rows which contain all null values
    # df.drop(null_idx, axis=0, inplace=True)


    df = df.reset_index(drop=True)

    # define features and predictors
    X = df.loc[:, 'message']
    Y = df.loc[:, 'related':]
    print("\nFeature space shape:", X.shape)
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
        n_estimators=10,
        # max_depth=4,
        max_features=0.8,
        max_samples=0.8,
        class_weight='balanced_subsample',
        n_jobs=N_JOBS,
        random_state=11
    )

    clf = LogisticRegression(**lg_params)
    # clf = RandomForestClassifier(**rf_params)



    # # build pipeline
    # pipeline = Pipeline([
    #     ('sent_vec', SentenceVector(size=50, min_count=10, iter=6, window=1)),
    #     # ('knn_imputer', KNNImputer(n_neighbors=5, weights='distance')),
    #     # ('imputer', SimpleImputer(strategy='mean')),
    #     # ('norm', QuantileTransformer(output_distribution='normal',
    #     #                               random_state=11)),
    #     ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    # ])


    count_vec = CountVectorizer(
    tokenizer=tokenize,
    ngram_range=(1, 1),
    # max_features=1000,
    max_df=0.95,
    min_df=0.01
    )

    sv = SentenceVector(size=100, min_count=5, iter=6, window=2)

    pipeline = Pipeline([

    ('features', FeatureUnion([

            ('count_vec_pipeline', Pipeline([
                    ('count_vect', count_vec),
                    # ('tfidf_tx', TfidfTransformer()),
                    ])),
            ('sent_vec_pipeline', Pipeline([
                    ('sentence_vec', sv),
                    ('quant_tx', StandardScaler()),
                    ])),
            ('keywords', KeywordSearch()),
            # ('sentence_vec', sv),

    ], n_jobs=1)),
    # ('norm', Normalizer(norm='l1', copy=False)),
    ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    ])

    # return grid search object
    return pipeline


def stacking_clf(params):

    sv = SentenceVector(size=100, min_count=8, window=1)

    multi_clf = MultiOutputClassifier

    estimators = [
        ('rf', make_pipeline(
            sv,
            RandomForestClassifier(params)
            )),

        ('log_reg', make_pipeline(
            sv,
            StandardScaler(),
            LogisticRegression()
            ))

        ]

    final_est = make_pipeline(
        # sv,
        LogisticRegression()
        )

    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        stack_method='predict'
        )

    return multi_clf(clf, n_jobs=6)

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
    print('Test Data Results:')
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


def train_model():

    pass

def main(sample_int=5000, grid_search=False):
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

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y.values,
                                                        test_size=0.15)

    show_info(X_train, y_train)
    del X, Y, database_filepath, model_filepath
    gc.collect()

    model = build_model()
    # model = stacking_clf(rf_params)

    start_time = time.perf_counter()
    print('\nTraining model...')
    model.fit(X_train.ravel(), y_train)
    end_time = time.perf_counter()
    print('\nTraining time:', np.round((end_time - start_time)/60, 4), 'min')

    print('\nEvaluating model...')
    gc.collect()
    evaluate_model(model, X_test.ravel(), y_test, category_names)

    # print('\nBest model:', model.best_estimator_)
    if grid_search:
        model = grid_search(model)
    if hasattr(model, 'best_params_'):
        print('\nBest params:', model.best_params_)
        print('\nBest score:', model.best_score_)
        print('Mean scores:', model.cv_results_['mean_test_score'])

    print('\nCross-validating...\n')
    gc.collect()
    scores = cross_val_score(
        model,
        X_train.ravel(),
        y_train,
        scoring='f1_weighted',
        cv=6,
        n_jobs=-1)
    print('\nCross-val mean score:\n', np.round(np.mean(scores), 4))
    print("-"*75)

    # print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
    # save_model(model, model_filepath)
    # print('\nTrained model saved!\n')


#%%
if __name__ == '__main__':
    main(sample_int=20000)