# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:32:00 2019

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

#%%

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
def drop_class(Y):
    """
    Checks amount of classes in each category.
    Drops class(es) (inplace) where there is less than 2 classes present.

    This functions does not return anything.
    """
    # extract category which has less than two classes
    print('Dropping class(es):', Y.nunique()[Y.nunique() < 2].index.tolist())
    # drop category, `child_alone`
    Y.drop(Y.nunique()[Y.nunique() < 2].index.tolist(), axis=1, inplace=True)

def load_data(database_filepath):
    """
    Import data from database into a DataFrame. Split DataFrame into
    features and predictors, `X` and `Y`.

    Preprocess data.

    Params:
        database_filepath: file path of database

    Returns:
        pd.DataFrame of features and predictors, `X` and `Y`, respectively.
    """
    # load data from CSV
    df = pd.read_csv(database_filepath)

    # explore `related` feature where its labeled as a `2`
    related_twos = df[df['related'] == 2]
    df.drop(index=related_twos.index, inplace=True)

    # define features and predictors
    X = df.loc[:, ['message']]
    Y = df.loc[:, 'related':]
    drop_class(Y)
    category_names = Y.columns.to_list()

    return X, Y, category_names

#%%


class Dense(BaseEstimator, TransformerMixin):
    """Convert sparse matrix into dense"""
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.toarray()



class SplitNote(BaseEstimator, TransformerMixin):
    """Split message at the word `Note`, keep message prior to `Note`"""
    def split_note(self, string):
        if 'Note' in string:
            return re.split('Note', string)[0]
        return string

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(self.split_note).values


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        try:
            sentence_list = nltk.sent_tokenize(text)
            for sentence in sentence_list:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            return False
        except Exception:
            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


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

#%%

class GetVerbNounCount(BaseEstimator, TransformerMixin):
    """
    Counts occurance of verbs and nouns within given text.
    Input can consist of a single string or an array of text
    documents from which to extract the counts of nouns and verbs.

    """

    def get_count_vb_nn(self, text):
        """
        Given input text, extracts the count of nouns and verbs.

        Returns:
        --------
            pd.Series of counts.
        """

        # EXTRACT `VERB` and `NOUN`
        # extract and count tags
        tags = []
        for tup in pos_tag(tokenize(text)):
            tags.append(tup[1])

        verb_tag = ['VB', 'VBN', 'VBD', 'VBP', 'VBZ', 'VBG']
        noun_tag = ['NNP', 'NN']
        # make series, check for occurance, convert dtype, and sum
        verb_count = pd.Series(tags).isin(verb_tag).astype(np.int32).sum()
        noun_count = pd.Series(tags).isin(noun_tag).astype(np.int32).sum()

        # concat
        return pd.Series([verb_count, noun_count])

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        Transforms input `X` into a pd.Series of noun and verb counts
        within given text.


        Parameters
        ----------
            X: np.array
                Array of text documents from which to extract counts
                of nouns and verbs.

        Returns
        -------
            pd.Series of counts of [verb, noun]

        """
        return pd.Series(X).apply(self.get_count_vb_nn)


#%%


class EntityCount(BaseEstimator, TransformerMixin):
    """
    Extracts various named entities from given text input.
    Input can consist of a single string or an array of text
    documents from which to extract the counts of named
    entities.

    """

    def get_entity(self, text):
        """
        Search and tag for Named Entities in message.

        Returns array representing found entities.
        ['S', 'GPE', 'ORGANIZATION']

        Returns:
        --------
            np.array

        """
        tree = ne_chunk(pos_tag(word_tokenize(text)))
        ne_list = ['GPE', 'PERSON', 'ORGANIZATION']
        ne_labels = []
        for item in tree.subtrees():
            ne_labels.append(item.label())

        return pd.Series(ne_list).isin(ne_labels).astype(np.int32)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(self.get_entity)



#%%



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





