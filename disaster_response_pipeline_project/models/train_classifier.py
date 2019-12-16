import nltk
#nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger',
#               'maxent_ne_chunker', 'words', 'word2vec_sample'])

from joblib import dump, load
import pickle
import sys
import re
import numpy as np
import pandas as pd
import time

from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from nltk import ne_chunk, pos_tag

from sklearn import svm
from sklearn.linear_model import (LogisticRegression,
                                  RidgeClassifier)

from sklearn.ensemble import (RandomForestClassifier,
                              BaggingClassifier,
                              RandomTreesEmbedding
                              )

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier


from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (StandardScaler, RobustScaler, Normalizer,
                                   FunctionTransformer, QuantileTransformer,
                                   PowerTransformer, OneHotEncoder)

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer,
                                             HashingVectorizer
                                             )

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import TruncatedSVD

from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)

from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, classification_report,
                             roc_curve, auc, accuracy_score, make_scorer)

from sklearn.utils import resample

from models.custom_tx import (StartingVerbExtractor,
                              KeywordSearch,
                              EntityCount,
                              GetVerbNounCount,
                              tokenize,
                              Dense
                              )

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
        scoring='f1_weighted',
        n_jobs=-1,
    )
    grid_cv.fit(X_train.ravel(), y_train)
    return grid_cv

def build_model():

    svc_params = dict(
        C = 2,
        kernel = 'linear',
        cache_size = 1000,
        class_weight = 'balanced',
        random_state = 11

    )

    # initialize classifier
    clf = svm.SVC(**svc_params)

    pipeline = Pipeline([
        ('count_vect', CountVectorizer(
                tokenizer=tokenize,
                ngram_range=(1, 2),
                max_features=300
                )),
        ('tfidf_tx', TfidfTransformer()),
        ('clf', MultiOutputClassifier(clf, n_jobs=-1))
    ])

    return pipeline




def evaluate_model(model, x_test, y_test, y_pred, category_names):
    y_pred = model.predict(x_test)
    # print label and f1-score for each
    avg = 'weighted'
    f1 = []
    prec = []
    rec = []
    acc = []
    #train_scores = []
    for i in range(y_test[:, :].shape[1]):
        f1.append(f1_score(y_test[:, i], y_pred[:, i], average=avg))
        acc.append(accuracy_score(y_test[:, i], y_pred[:, i]))
        rec.append(recall_score(y_test[:, i], y_pred[:, i], average=avg))
        prec.append(precision_score(y_test[:, i], y_pred[:, i], average=avg))

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
    print(f1_df.agg(['mean', 'median', 'std']))
    print('='*75)
    print('\n')




def save_model(model, filepath):
    """


    Parameters
    ----------
        model : estimator

        filepath : save model to this directory

    Returns
    -------
        None.

    """
    try:
        dump(model, filepath)
    except Exception as e:
        print(e)
        print('Failed to pickle model')



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