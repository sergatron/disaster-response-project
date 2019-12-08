import sys


# import libraries
import re
import numpy as np
import pandas as pd

import sqlite3
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from nltk import ne_chunk, pos_tag

from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, FunctionTransformer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, FeatureHasher
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, make_scorer

from sklearn.utils import resample

#%%

def load_data(database_filepath, file_type='db'):
    """
    Import data from database into a DataFrame. Split DataFrame into
    features and predictors, `X` and `Y`.

    Params:
        database_filepath: file path of database

    Returns:
        pd.DataFrame of features and predictors, `X` and `Y`, respectively.
    """
    if file_type == 'db':
        # load data from database
        engine = create_engine(f'sqlite:///{database_filepath}.db')
        df = pd.read_sql_table(f'{database_filepath}', engine)

        # define features and predictors
        X = df.loc[:, ['message']]
        Y = df.loc[:, 'related':]
        category_names = Y.columns.to_list()

        return X, Y, category_names

    # load data from CSV
    df = pd.read_csv(f'{database_filepath}.csv')

    # define features and predictors
    X = df.loc[:, ['message']]
    Y = df.loc[:, 'related':]
    category_names = Y.columns.to_list()

    return X, Y, category_names


def tokenize(text):
    """

    """

    # load stopwords
    stop_words = stopwords.words("english")

    # remove punctuations (retain alphabetical and numeric chars) and convert to all lower case
    # tokenize resulting text
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", ' ', text.lower().strip()))

    # lemmatize and remove stop words
    lemmatized = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]

    return lemmatized


#%%

class SplitNote(BaseEstimator, TransformerMixin):
    """Split message at the word `Note`, keep message prior to `Note`"""
    def split_note(self, string):
        return re.split('Note', string)[0]

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(pd.Series(X).apply(self.split_note))


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

#%%

#X, y = load_data('data/disaster_message_cat')
#
#SplitNote().fit_transform(X.values.ravel())
#
#StartingVerbExtractor().fit_transform(X.values.ravel())

#%%

def grid_search(model, params, X_train=X_train, y_train=y_train):

    grid_cv = GridSearchCV(
        model,
        params,
        cv=3,
        n_jobs=-1,
    )
    grid_cv.fit(X_train.ravel(), y_train)
    return grid_cv

def build_model(clf, params):

    svc_params = dict(
        C = 1,
        kernel = 'sigmoid',
        cache_size = 2000,
        class_weight = 'balanced',
        random_state = 11

    )

    # initialize classifier
    clf = svm.SVC(**svc_params)

    pipeline = Pipeline([
        ('preprocess', SplitNote()),

        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([

                ('count_vect', CountVectorizer(tokenizer=tokenize,
                                               ngram_range=(1,2),
                                               max_df=0.9,
                                               min_df=0.02,
                                               max_features=500
                                              )
                ),
                ('tfidf_tx', TfidfTransformer()),
                ('decomp', TruncatedSVD(n_components=10, random_state=11))
            ])),
            ('extract_verb', StartingVerbExtractor()),

        ], n_jobs=-1)),

        ('clf', MultiOutputClassifier(clf, n_jobs=-1))
    ])
    return pipeline




def evaluate_model(model, x_test, y_test, y_pred, category_names):

    # print label and f1-score for each
    scores = []
    for i in range(y_test[:, :].shape[1]):
        scores.append(f1_score(y_test[:, i], y_pred[:, i]))
    for lbl, scr in zip(category_names, scores):
        print(lbl, ':', np.round(scr, 4))

    print('\n')
    print('Average across all labels:', sum(scores) / len(scores))
    print('\n')

    # summarize f1-scores in DF
    f1_df = pd.DataFrame({'f1-score': scores}, index=category_names)
    return f1_df


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()