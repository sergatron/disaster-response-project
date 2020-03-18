import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore')

import gc

from joblib import dump, load
import pickle


import re
import numpy as np
import pandas as pd
import time

import nltk
# nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger',
#               'maxent_ne_chunker', 'words', 'word2vec_sample'])


from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer
# from nltk import ne_chunk, pos_tag


import lightgbm as lgb

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

# from sklearn.compose import ColumnTransformer
# from sklearn.base import BaseEstimator, TransformerMixin

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
    tokens = word_tokenize(re.sub(r"[^a-zA-Z]", ' ', text.lower().strip()))

    lemm = WordNetLemmatizer()
    # lemmatize and remove stop words
    lemmatized = [lemm.lemmatize(word) for word in tokens if word not in stop_words]

    return lemmatized


# In[4]:
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

    Y.drop(Y.nunique()[Y.nunique() < 2].index.tolist(), axis=1, inplace=True)

    # extract label names
    category_names = Y.columns.to_list()

    return X, Y, category_names

#%%

def grid_search(model, grid_params):
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
    N_JOBS = 4
    bg_params = dict(
        n_estimators=60,
        n_jobs=N_JOBS,
        random_state=11
        )
    ext_params = dict(
        n_estimators=300,
        n_jobs=N_JOBS,
        random_state=11
        )
    rf_params = dict(
        n_estimators=400,
        max_depth=12,
        class_weight='balanced',
        n_jobs=N_JOBS,
        random_state=11
        )

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

    #clf = MLPClassifier(random_state=11,
    #                    hidden_layer_sizes=(400,),
    #                    alpha=0.01,
    #                    learning_rate='adaptive',
    #                    activation='relu',
    #                    max_iter=500)
    # clf = LogisticRegression(**lg_params)
    # clf = BaggingClassifier(**bg_params)
    clf = RandomForestClassifier(**rf_params)
    # clf = ExtraTreesClassifier(**ext_params)

    count_vec = CountVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        # max_features=1000,
        max_df=0.95,
        min_df=0.01
        )
    hash_vec = HashingVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        n_features=200,
        )

    # build pipeline
    pipeline = Pipeline([
        ('vectorizer', count_vec),
        ('tfidf_tx', TfidfTransformer()),
        ('norm', Normalizer(norm='l1', copy=False)),


        # ('decomp', TruncatedSVD(n_components=4,
        #                         random_state=11)),
        # ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
        ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    ])

    # pipeline = Pipeline([

    # ('features', FeatureUnion([
    #         ('text_pipeline', Pipeline([
    #                 ('count_vect', count_vec)
    #                 ])),

    #         # ('keywords', KeywordSearch()),
    #         # ('entity_count', EntityCount()),
    #         # ('verb_noun_count', GetVerbNounCount()),
    #         ('sentence_vec', SentenceVector()),


    # ], n_jobs=1)),

    # ('tfidf_tx', TfidfTransformer()),
    # ('norm', Normalizer(norm='l1', copy=False)),
    # ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    # ])

    # return grid search object
    return pipeline


def lightgbm_model(params):
    N_JOBS = 6

    count_vec = CountVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        # max_features=200
        )

    clf = lgb.LGBMClassifier(**params)

    # # build pipeline
    pipeline = Pipeline([
        ('vectorizer', count_vec),
        ('tfidf_tx', TfidfTransformer()),
        ('norm', Normalizer(norm='l1', copy=False)),


        # ('decomp', TruncatedSVD(n_components=3,
        #                         random_state=11)),
        # ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
        ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    ])

    #pipeline = Pipeline([

    #('features', FeatureUnion([
    #        ('text_pipeline', Pipeline([
    #                ('count_vect', count_vec)
    #                ])),

    #        ('keywords', KeywordSearch()),
            # ('entity_count', EntityCount()),
            # ('verb_noun_count', GetVerbNounCount()),
            # ('verb_extract', StartingVerbExtractor()),


    #], n_jobs=1)),

    #('tfidf_tx', TfidfTransformer()),
    # ('poly', PolynomialFeatures(degree=2, interaction_only=False)),
    #('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    #])

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
def main(sample_int=5000, grid_search=False, LGBM=False):
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
    params = {
    # "boosting": "gbdt",
    "num_leaves": 1000,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "pos_bagging_fraction": 0.9,
    "neg_bagging_fraction": 0.2,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "metric": 'binary_logloss',
    "unbalance": True,
    # "reg_lambda": 3,
    "random_state": 11,
    "n_jobs": 4,
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        database_filepath = "data/disaster_response.db"
        model_filepath = "models/disaster_clf.pkl"


        print('\nLoading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath, n_sample=sample_int)
        X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                            Y.values,
                                                            test_size=0.2)

        show_info(X_train, y_train)
        del X, Y, database_filepath, model_filepath
        gc.collect()

        if LGBM:
            print('\n --- Using LightGBM --- \n')
            start_time = time.perf_counter()
            print('\nTraining model...')
            model = lightgbm_model(params)
            model.fit(X_train, y_train,
                      #  eval_set=[(X_test, y_test)],
                      #  eval_metric='binary_logloss',
                      #  early_stopping_rounds=25
                       )
            end_time = time.perf_counter()
            print('\nTraining time:', np.round((end_time - start_time)/60, 4), 'min')


        else:
            model = build_model()
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
            cv=3,
            n_jobs=-1)
        print('\nCross-val mean score:\n', np.round(np.mean(scores), 4))
        print("-"*75)

        # print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
        # save_model(model, model_filepath)
        # print('\nTrained model saved!\n')



if __name__ == '__main__':
    main(sample_int=18000, LGBM=False)
