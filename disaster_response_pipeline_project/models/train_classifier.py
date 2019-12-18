import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore')

from joblib import dump, load

import pickle
import sys
import re
import numpy as np
import pandas as pd
import time

import nltk
nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger',
              'maxent_ne_chunker', 'words', 'word2vec_sample'])


from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import ne_chunk, pos_tag

from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (RandomForestClassifier,
                              BaggingClassifier,
                              RandomTreesEmbedding,
                              GradientBoostingClassifier
                              )

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, QuantileTransformer

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

# from custom_tx import tokenize

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
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    # load stopwords
    stop_words = stopwords.words("english")

    # remove additional words
    remove_words = ['one', 'reason', 'see']
    for addtl_word in remove_words:
        stop_words.append(addtl_word)

    # remove punctuations (retain alphabetical and numeric chars) and convert to all lower case
    # tokenize resulting text
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", ' ', text.lower().strip()))
    lemm = WordNetLemmatizer()
    # lemmatize and remove stop words
    lemmatized = [lemm.lemmatize(word) for word in tokens if word not in stop_words]

    return lemmatized


# In[4]:
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
    engine = create_engine('sqlite:///data/disaster_response.db')
    df = pd.read_sql_table('disaster_response', engine)

    #           *** TEMPORARY SAMPLE TO TEST SCRIPT ***
    df = df.sample(3000)

    # explore `related` feature where its labeled as a `2`
    related_twos = df[df['related'] == 2]
    df.drop(index=related_twos.index, inplace=True)

    df = df.reset_index(drop=True)

    # define features and predictors
    X = df.loc[:, 'message']
    Y = df.loc[:, 'related':]


    # extract label names
    category_names = Y.columns.to_list()

    return X, Y, category_names

#%%

def grid_search(model):
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    bg = BaggingClassifier()

    grid_params = [
        # {
        #     'clf__estimator': [rf],
        #     'clf__estimator__n_estimators': [120],
        #     # 'clf__estimator__max_depth': [3],
        #     'count_vec__max_features': [50, 100, 300],
        #     },

        {
            'clf__estimator': [bg],
            'clf__estimator__n_estimators': [60],
            'clf__estimator__max_samples': [1.0],
            'clf__estimator__n_jobs': [-1],
            'count_vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
            }
        ]


    grid_cv = GridSearchCV(
        model,
        grid_params,
        cv=3,
        scoring='f1_weighted',
        n_jobs=1,
    )

    return grid_cv

def build_model():


    rf_params = dict(
        n_estimators=40,
        max_depth=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=11
        )
    # clf = svm.SVC(**svc_params)
    clf = RandomForestClassifier(**rf_params)

    count_vec = CountVectorizer(
                tokenizer=tokenize,
                ngram_range=(1, 2),
                )
    tfidf = TfidfTransformer()

    # build pipeline
    pipeline = Pipeline([
        ('count_vect', count_vec),
        ('tfidf_tx', tfidf),
        ('decomp', TruncatedSVD(n_components=2,
                                random_state=11)),
        ('clf', MultiOutputClassifier(clf, n_jobs=-1))
    ])

    # return grid search object
    return grid_search(pipeline)




def evaluate_model(model, x_test, y_test, category_names):
    y_pred = model.predict(x_test)
    # print label and f1-score for each
    avg = 'weighted'
    f1 = []
    prec = []
    rec = []
    acc = []
    for i in range(y_test[:, :].shape[1]):
     #    with warnings.catch_warnings():
    	# # ignore all caught warnings
     #        warnings.filterwarnings("ignore")

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
    print(f1_df.agg(['mean', 'median', 'std']))
    print('='*75)
    print('\n')



def show_info(X_train, y_train):
    print("X-shape:", X_train.shape)
    print("Y-shape:", y_train.shape)

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


#%%
def main():
    if len(sys.argv) == 3:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            database_filepath, model_filepath = sys.argv[1:]

            print('\nLoading data...\n    DATABASE: {}'.format(database_filepath))

            X, Y, category_names = load_data(database_filepath)
            X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                                Y.values,
                                                                stratify=Y['floods'].values,
                                                                test_size=0.2)

            show_info(X_train, y_train)

            print('\nBuilding model...')
            model = build_model()

            start_time = time.perf_counter()
            print('\nTraining model...')
            model.fit(X_train.ravel(), y_train)
            end_time = time.perf_counter()

            print('\nEvaluating model...')
            evaluate_model(model, X_test.ravel(), y_test, category_names)

            print('\nBest model:', model.best_estimator_)
            print('Best params:', model.best_params_)

            print('\nTraining time:', np.round((end_time - start_time)/60, 4), 'min')

            print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('\nTrained model saved!\n')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()