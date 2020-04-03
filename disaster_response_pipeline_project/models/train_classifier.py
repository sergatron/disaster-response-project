import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore')

from joblib import dump

import gc
import sys
import re
import numpy as np
import pandas as pd
import time

import nltk
# nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger',
#               'maxent_ne_chunker', 'words', 'word2vec_sample'])

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             HashingVectorizer)

from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, classification_report,
                             accuracy_score)

from sklearn.utils import resample

from custom_transform import (KeywordSearch, StartingVerbExtractor,
                              GetVerbNounCount, EntityCount)

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
                    'we', 'us', 'you', 'me', 'their', 'there', 'here']
    for addtl_word in remove_words:
        stop_words.append(addtl_word)

    # remove punctuations (retain alphabetical and numeric chars) and convert to all lower case
    # tokenize resulting text
    tokens = word_tokenize(re.sub(r"[^a-zA-Z]", ' ', text.lower().strip()))

    # drop stop words
    no_stops = [word for word in tokens if word not in stop_words]

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
    if n_sample > 0:
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

#%%

def grid_search(model, X, y):
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
    N_JOBS = 2
    # ext = ExtraTreesClassifier()
    # bg = BaggingClassifier()
    rf = RandomForestClassifier()
    lg = LogisticRegression()
    svc = svm.SVC()

    grid_params = [

        {
            'clf__estimator': [rf],
            'clf__estimator__n_estimators': [20, 50],
            'clf__estimator__max_depth': [4, 8],
            'clf__estimator__max_samples': [0.6, 0.8],
            'clf__estimator__max_features': [0.6, 0.8],
            'clf__estimator__n_jobs': [N_JOBS],
            'clf__estimator__random_state': [11]
            },
        {
            'clf__estimator': [lg],
            'clf__estimator__C': [0.1, 0.01],
            'clf__estimator__penalty': ['l2'],
            'clf__estimator__solver': ['newton-cg'],
            'clf__estimator__class_weight': ['balanced'],
            'clf__estimator__multi_class': ['auto'],
            'clf__estimator__n_jobs': [-1],
            'clf__estimator__random_state': [11]
            },
        # {
        #     'clf__estimator': [svc],
        #     'clf__estimator__C': [0.1, 0.01, 0.05],
        #     'clf__estimator__kernel': ['rbf'],
        #     'clf__estimator__gamma': [0.02],
        #     'clf__estimator__class_weight': ['balanced'],
        #     'clf__estimator__random_state': [11]
        #     }

        ]

    print('\nTuning hyper-parameters...')
    grid_cv = GridSearchCV(
        model,
        grid_params,
        cv=3,
        scoring='recall_weighted',
        n_jobs=N_JOBS,
    )

    grid_cv.fit(X, y)

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

    rf_params = dict(
        n_estimators=100,
        max_depth=8,
        max_features=0.8,
        max_samples=0.8,
        class_weight='balanced',
        n_jobs=N_JOBS,
        random_state=11
        )
    lg_params = dict(
        C = 0.1,
        solver = 'newton-cg',
        penalty = 'l2',
        class_weight = 'balanced',
        multi_class = 'auto',
        n_jobs = N_JOBS,
        random_state = 11
    )

    svc_params = dict(
        C = 0.01,
        kernel = 'rbf',
        gamma = 0.02,
        cache_size = 1000,
        class_weight = 'balanced',
        random_state = 11
        )

    # clf = RandomForestClassifier(**rf_params)
    clf = LogisticRegression(**lg_params)
    # clf = svm.SVC(**svc_params)

    count_vec = CountVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        dtype=np.uint16,
        max_features=2500,
        max_df=0.98,
        min_df=5
        )
    hash_vec = HashingVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 1),
        n_features=200,
        dtype=np.float32
        )

    pipeline = Pipeline([

    ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                    ('count_vect', count_vec),
                    ('tfidf_tx', TfidfTransformer()),
                    ])),
            ('keywords', KeywordSearch()),
            # ('verb_noun_count', GetVerbNounCount()),
            # ('entity_count', EntityCount()),
            # ('verb_extract', StartingVerbExtractor()),

    ], n_jobs=1)),

    # ('norm', Normalizer()),
    ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))])

    # return grid search object
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

    if isinstance(y_test, (pd.DataFrame, pd.Series)):
        y_test = y_test.values

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
    print("\nX-shape:", X_train.shape)
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

def upsample(X_train, y_train, target_col_name, sample_fraction=0.25):

    # combine train sets
    X_c = pd.concat([X_train, y_train], axis=1)
    # extract `success` and `fail` instances, `success` represented by 1
    fail = X_c[X_c[target_col_name] == 0]
    success = X_c[X_c[target_col_name] == 1]

    # upsample to match 'fail' class
    success_upsampled = resample(success,
                                  replace=True,
                                  n_samples=int(len(fail)*(sample_fraction)),
                                  random_state=11
                                  )
    # put back together resample `success` and fail
    upsample = pd.concat([fail, success_upsampled])
    # split back into X_train, y_train
    X_train = upsample['message']
    y_train = upsample.drop('message', axis=1)

    return X_train, y_train


def downsample(X_train, y_train, target_col_name, sample_fraction=1.0):
    """

    Parameters
    ----------
        X_train : pd.DataFrame
            Training feature space subset.

        y_train : pd.DataFrame
            Training target variable subset.

        target_col_name : str
            Target variable to resample.

        sample_fraction : float, optional
            Controls the number of samples being drawn from an array.
            This essentially controls the magnitude of downsampling. Increasing
            this value will draw more samples with a positive instance
            from array thus chaning affecting the balance.

    Returns
    -------
        X_train : pd.DataFrame
        y_train : pd.DataFrame

    """
    # combine train sets
    X_c = pd.concat([X_train, y_train], axis=1)
    # extract `success` and `fail` instances, `success` represented by 1
    fail = X_c[X_c['aid_related'] == 0]
    success = X_c[X_c['aid_related'] == 1]

    # downsample w/replacment; number of samples = len(fail)
    # this essentially add more instances of `fail` to improve balance
    success_downsampled = resample(fail,
                                  replace=True,
                                  n_samples=int(len(success)*sample_fraction),
                                  random_state=11
                                  )

    # put back together resample `success` and fail
    downsample = pd.concat([success, success_downsampled])
    # split back into X_train, y_train
    X_train = downsample['message']
    y_train = downsample.drop('message', axis=1)

    return X_train, y_train


#%%
def main(sample_int=5000, gs=False, cv_split=3):
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

            X, Y, category_names = load_data(database_filepath, n_sample=sample_int)
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                Y,
                                                                test_size=0.2)
            del X, Y, database_filepath
            gc.collect()

            show_info(X_train, y_train)

            #### Upsample more important features
            for col in ['missing_people', 'clothing', 'food']:
                X_train, y_train = upsample(X_train, y_train, col, 0.9)
            #### Downsample
            X_train, y_train = downsample(X_train, y_train, 'aid_related', 0.4)
            print('\nResampled shape:')
            show_info(X_train, y_train)


            model = build_model()
            print('Model params:\n', model.get_params()['clf__estimator'])
            start_time = time.perf_counter()
            print('\nTraining model...')
            model.fit(X_train.ravel(), y_train)
            end_time = time.perf_counter()
            print('\nTraining time:', np.round((end_time - start_time)/60, 4), 'min')

            print('\nEvaluating model...')
            evaluate_model(model, X_test.ravel(), y_test, category_names)


            # perform grid-search if specified
            if gs:
                grid_cv = grid_search(model, X_train, y_train)
                if hasattr(grid_cv, 'best_params_'):
                    print('\nBest params:', grid_cv.best_params_)
                    print('\nBest score:', grid_cv.best_score_)
                    print('Mean scores:', grid_cv.cv_results_['mean_test_score'])
                model = grid_cv.best_estimator_
                print('\nEvaluating best estimator...')
                evaluate_model(model, X_test.ravel(), y_test, category_names)

            del X_test, y_test, category_names
            gc.collect()

            if cv_split > 0:
                # perform cross-validation if the number of splits > 0
                print('-'*75)
                print('\nCross-validating...\n')
                kf = KFold(n_splits=cv_split, shuffle=True, random_state=11)
                scores = cross_val_score(
                    model,
                    X_train.ravel(),
                    y_train,
                    scoring='recall_weighted',
                    cv=kf,
                    n_jobs=-1)
                print('\nCross-val scores:\n', scores)
                print('-'*75)

            print('\nFinal model params:\n', model.get_params()['clf__estimator'])

            print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('\nTrained model saved!\n')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main(sample_int=0, gs=False, cv_split=3)
