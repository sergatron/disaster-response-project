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
# nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger',
#               'maxent_ne_chunker', 'words', 'word2vec_sample'])


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
                              ExtraTreesClassifier,
                              BaggingClassifier,
                              RandomTreesEmbedding,
                              StackingClassifier
                              )



from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (Normalizer, QuantileTransformer,
                                   PolynomialFeatures)

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
                             accuracy_score, make_scorer, log_loss)

from sklearn.utils import resample

from sklearn.neural_network import MLPClassifier

from custom_transform import (KeywordSearch, StartingVerbExtractor,
                              GetVerbNounCount, EntityCount)

#%%

def tokenize(text):
    """

    Applies the following steps to process input `text`.
    1. Replace `url` with empty space.
    2. Remove stopwords.
    3. Tokenize and lemmatize input `text`.
    4. Converts to lower case and strips whitespaces.

    Params:
    -------
        text: str
            string to process by applying above steps

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


    # remove punctuations (retain alphabetical and numeric chars) and convert
    # to all lower case tokenize resulting text
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", ' ', text.lower().strip()))

    lemm = WordNetLemmatizer()
    # lemmatize and remove stop words
    lemmatized = [lemm.lemmatize(word) for word in tokens if word not in stop_words]

    return lemmatized


# In[4]:
def load_data(database_filepath):
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
    df = df.sample(5000)

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
    N_JOBS = 1
    ext = ExtraTreesClassifier(n_estimators=10,
                               n_jobs=6,
                               class_weight='balanced')
    bg = BaggingClassifier(n_jobs=6)

    grid_params = [
        # {
        #     'clf__estimator': [ext],
        #     'clf__estimator__n_estimators': [100],
        #     # 'clf__estimator__max_depth': [None, 2, 3],
        #     # 'clf__estimator__bootstrap': [True, False],
        #     'clf__estimator__n_jobs': [6],
        #     'clf__estimator__random_state': [11]
        #     },

        {
            'clf__estimator': [bg],
            'clf__estimator__n_estimators': [100, 110, 120],
            # 'decomp__n_components': [2, 3],
            # 'clf__estimator__max_samples': [0.7, 0.8, 1.0],
            # 'clf__estimator__max_features': [0.7, 0.8, 1.0],
            # 'clf__estimator__bootstrap_features': [True, False],
            # 'clf__estimator__bootstrap': [True, False],
            # 'vectorizer__ngram_range': [(1,2), (1,3), (2,2)],
            'clf__estimator__n_jobs': [6],
            'clf__estimator__random_state': [11]
            }
        ]


    grid_cv = GridSearchCV(
        model,
        grid_params,
        cv=3,
        scoring='f1_weighted',
        n_jobs=N_JOBS,
    )

    return grid_cv

def build_model():
    """

    Creates a Pipeline object with preset initial params for estimators
    and classifier.

    Returns:
    -------
        Pipeline object

    """
    N_JOBS = -1
    bg_params = dict(
        n_estimators=60,
        n_jobs=N_JOBS,
        random_state=11
        )
    ext_params = dict(
        n_estimators=105,
        n_jobs=N_JOBS,
        random_state=11
        )
    rf_params = dict(
        n_estimators=140,
        # max_depth=2,
        class_weight='balanced',
        n_jobs=N_JOBS,
        random_state=11
        )
    rt_params = dict(
        n_estimators=30,
        max_depth=3,
        n_jobs = N_JOBS,
        random_state = 11
        )

    clf = MLPClassifier(random_state=11,
                        hidden_layer_sizes=(200,),
                        alpha=0.00001,
                        learning_rate='adaptive',
                        activation='relu',
                        max_iter=300)

    # clf = BaggingClassifier(**bg_params)
    # clf = RandomForestClassifier(**rf_params)
    # clf = ExtraTreesClassifier(**ext_params)

    count_vec = CountVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        # max_features=200
        )
    hash_vec = HashingVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        n_features=200,
        )

    # # build pipeline
    # pipeline = Pipeline([
    #     ('vectorizer', count_vec),
    #     ('tfidf_tx', TfidfTransformer()),
    #     # ('norm', Normalizer(norm='l2', copy=False)),


    #     ('decomp', TruncatedSVD(n_components=3,
    #                             random_state=11)),
    #     # ('poly', PolynomialFeatures(degree=10, interaction_only=False)),
    #     ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    # ])

    pipeline = Pipeline([

    ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                    ('count_vect', count_vec)
                    ])),

            ('keywords', KeywordSearch()),
            ('verb_noun_count', GetVerbNounCount()),
            # ('entity_count', EntityCount()),
            # ('verb_extract', StartingVerbExtractor()),


    ], n_jobs=1)),

    ('tfidf_tx', TfidfTransformer()),
    ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    ])


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


#%%
def main():
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
    if len(sys.argv) == 3:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            database_filepath, model_filepath = sys.argv[1:]

            print('\nLoading data...\n    DATABASE: {}'.format(database_filepath))

            X, Y, category_names = load_data(database_filepath)
            X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                                Y.values,
                                                                test_size=0.2)

            show_info(X_train, y_train)

            print('\nBuilding model...')
            model = build_model()

            start_time = time.perf_counter()
            print('\nTraining model...')
            model.fit(X_train.ravel(), y_train)
            end_time = time.perf_counter()
            print('\nTraining time:', np.round((end_time - start_time)/60, 4), 'min')

            print('\nEvaluating model...')
            evaluate_model(model, X_test.ravel(), y_test, category_names)

            # print('\nBest model:', model.best_estimator_)
            if hasattr(model, 'best_params_'):
                print('\nBest params:', model.best_params_)
                print('\nBest score:', model.best_score_)
                print('Mean scores:', model.cv_results_['mean_test_score'])

            print('\nCross-validating...\n')
            scores = cross_val_score(
                model,
                X_train.ravel(),
                y_train,
                scoring='f1_weighted',
                cv=3,
                n_jobs=-1)
            print('\nCross-val scores:\n', scores)

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
