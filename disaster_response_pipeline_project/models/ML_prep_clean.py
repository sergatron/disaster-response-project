#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y


#%%
import nltk
#nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger',
#               'maxent_ne_chunker', 'words', 'word2vec_sample'])

# import libraries
import dill as pickle
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

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
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
                             roc_auc_score,accuracy_score, make_scorer)

from sklearn.utils import resample

from custom_transform import (StartingVerbExtractor,
                              KeywordSearch,
                              EntityCount,
                              GetVerbNounCount,
                              tokenize,
                              Dense,
                              SentenceVector
                              )



# In[3]:


pd.options.display.max_columns = 60

def drop_class(Y):
    """
    Checks distribution of classes in each category.
    Drops class(es) (inplace) where there is less than 2 classes present.

    For example, if one of the target classes contain only ones of zeros,
    that target class will be removed.

    This functions does not return anything.
    """
    # extract category which has less than two classes
    print('Dropping class(es):', Y.nunique()[Y.nunique() < 2].index.tolist())
    # drop category, `child_alone`
    Y.drop(Y.nunique()[Y.nunique() < 2].index.tolist(), axis=1, inplace=True)

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
    # load data from CSV
    df = pd.read_csv(database_filepath)

    # explore `related` feature where its labeled as a `2`
    related_twos = df[df['related'] == 2]
    df.drop(index=related_twos.index, inplace=True)

    df = df.reset_index(drop=True)

    # define features and predictors
    X = df.loc[:, ['message']]
    Y = df.loc[:, 'related':]
    drop_class(Y)
    category_names = Y.columns.to_list()

    return X, Y, category_names

#%%
# load data from database
engine = create_engine('sqlite:///data/disaster_response.db')
df = pd.read_sql_table('disaster_response', engine)


#idx = 78
#df.loc[idx, 'message':]
#msg = df.loc[idx, 'message']
#df.loc[idx, 'related':]
#print(msg)
#
#df[df['genre'] == 'news'].loc[13229, :]

#%%
X = df.loc[:, ['message']]
Y = df.loc[:, 'related':]


# explore `related` feature where its labeled as a `2`
related_twos = df[df['related'] == 2]

# try dropping the above rows
df.drop(index=related_twos.index, inplace=True)
df = df.reset_index(drop=True)
# check count of classes
df.nunique()

# now `related` has been reduced down to two classes


# In[8]:
# EXPLORE MESSAGES IN MORE DETAIL

idx = 9
df.loc[idx, 'message']
df.loc[idx, 'related':]

# all rows except `related` are equal to zero at given index
(df.loc[idx, 'related':] == 0).all()

# iterate over each message, find each row which contains ALL zeros
row_sum = df.loc[:, 'related':].apply(sum, axis=1)
drop_idx = row_sum[row_sum < 1].index
print(len(drop_idx))


# inspect indecies before dropping
# NOTE: This message is asking for FOOD AND WATER. However, ALL labels
#       indicate NO need for help
idx = drop_idx[78]
df.loc[idx, 'message']
df.loc[idx, 'message':]


idx = drop_idx[77]
df.loc[idx, 'message']
df.loc[idx, 'message':]


#%%
from spellchecker import SpellChecker

spell = SpellChecker()

# find those words that may be misspelled
misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    print(spell.candidates(word))

idx = 15
df.loc[idx, 'message']
df.loc[idx, 'related':]

#%%

### COMBINE OUTPUT CATEGORIES


# Combine Weather
(df['weather_related'] + df['other_weather']).unique()

df.loc[:, 'related':].sum(axis=1)


#%%

# FIND ROWS WITH NO POSITIVE INSTANCES

# where sum across entire row is less than 1
null_idx = np.where(df.loc[:, 'related':].sum(axis=1) < 1)[0]

# drop rows which contain all null values
df.drop(null_idx, axis=0, inplace=True)


#%%

# # CHECK BALANCE
# before = (df.loc[:, 'related':].sum() / df.loc[:, 'related':].shape[0]).sort_values()

# # DROP INDEX
# df.drop(index=drop_idx, inplace=True)

# # CHECK BALANCE, AGAIN
# after = (df.loc[:, 'related':].sum() / df.loc[:, 'related':].shape[0]).sort_values()

# np.c_[before, after]

# # REPLACE `related` with zeros
# df['related'].replace(to_replace=1, value=0, inplace=True)
# df['related'].sum()


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



#%%

idx = 99
msg = df.loc[idx, 'message']
df.loc[idx, 'related':]
print(msg)


# tokenize, pos tag, then recognize named entities in text
tree = ne_chunk(pos_tag(word_tokenize(msg)))
print(tree)

ne_list = ['GPE', 'PERSON', 'ORGANIZATION']
ne_labels = []
for item in tree.subtrees():
    ne_labels.append(item.label())

# FOUND ENTITIES
pd.Series(ne_list).isin(ne_labels).astype(np.int32).values


#%%
#print(nltk.FreqDist(lem).most_common())
#nltk.ConditionalFreqDist(pos_tag(lem))['is'].most_common()

# POSITION OF VERBS AND NOUNS



# In[30]:
N_JOBS = -1

# LogisticRegression params
lg_params = dict(
    C = 12,
    solver = 'newton-cg',
    penalty = 'l2',
    class_weight = 'balanced',
    multi_class = 'multinomial',
    n_jobs = N_JOBS,
    random_state = 11

)


svc_params = dict(
    C = 2,
    kernel = 'linear',
#    gamma = 0.002,
    cache_size = 1000,
    class_weight = 'balanced',
    random_state = 11

)

rt_params = dict(
        n_estimators=300,
        max_depth=10,
        n_jobs = N_JOBS,
        random_state = 11
        )
rf_params = dict(
        n_estimators=40,
        max_depth=4,
        # min_samples_split=10,
        class_weight='balanced',
        n_jobs=N_JOBS,
        random_state=11
        )

# define classifier
clf = LogisticRegression(**lg_params)
# clf = svm.SVC(**svc_params)
# clf = RandomForestClassifier(**rf_params)
#

# pipeline = Pipeline([
#         ('count_vect', CountVectorizer(
#                 tokenizer=tokenize,
#                 ngram_range=(1, 2),
# #                max_features=200
#                 )),
#         ('tfidf_tx', TfidfTransformer()),
#         ('clf', MultiOutputClassifier(clf, n_jobs=6))
#     ])

pipeline = Pipeline([

    ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                    ('count_vect', CountVectorizer(
                            tokenizer=tokenize,
                            ngram_range=(1, 1),
                            ))
                    ])),

            # ('keywords', KeywordSearch()),
            # ('verb_noun_count', GetVerbNounCount()),
            # ('entity_count', EntityCount()),
            # ('verb_extract', StartingVerbExtractor()),


    ], n_jobs=N_JOBS)),

    ('tfidf_tx', TfidfTransformer()),
    # ('quantile_tx', QuantileTransformer(output_distribution='normal',
    #                                     random_state=11)),
    # ('decomp', TruncatedSVD(n_components=2,
    #                         random_state=11)),
    # ('rt', RandomTreesEmbedding(**rt_params)),
    # ('dense', Dense()),
    # ('poly', PolynomialFeatures(degree=3, interaction_only=True)),
    # ('scale', RobustScaler(with_centering=False)),
    ('clf', MultiOutputClassifier(clf, n_jobs=N_JOBS))
    ])

# use ColumnTransfomer to combine transformations
# NOTE:
#    OneHot expects 2-D, therefore, the column(s) must be specified
#    as a list!
#full_pipe = Pipeline([
#        ('union', ColumnTransformer([
#                ('category', OneHotEncoder(), [0]),
#                ('messages', pipeline, 1),
#                ])),
#        ('clf', MultiOutputClassifier(clf,n_jobs=-1))
#        ], memory='models/cache')

# In[56]:

# RESET INDEX
df.reset_index(drop=True, inplace=True)
df['genre'] = df['genre'].astype('category')

X = df.loc[:, 'message']
Y = df.loc[:, 'related':]

# DEFINE `X` AND `Y` AGAIN
sample_it = True
n_samples = 10000
if sample_it:
    sampler = df.sample(n_samples)
    X = sampler.loc[:, 'message']
    Y = sampler.loc[:, 'related':]

print('X-shape:', X.shape)
print('Y-shape:', Y.shape)

# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[31]:


# extract category which has less than two classes
print(Y.nunique()[Y.nunique() < 2].index.tolist())

# drop category, `child_alone`
Y.drop(Y.nunique()[Y.nunique() < 2].index.tolist(), axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    Y.values,
                                                    # stratify=Y['offer'].values,
                                                    test_size=0.15)


# In[33]:
print('Training model...')

start_time = time.perf_counter()

pipeline.fit(X_train.ravel(), y_train)
y_pred = pipeline.predict(X_test.ravel())

#full_pipe.fit(X_train, y_train)
#y_pred = full_pipe.predict(X_test)

end_time = time.perf_counter()

print('\n')
print('-'*75)
print('Training time:', np.round((end_time - start_time)/60, 4), 'min')
print('\n')

# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

print('Scoring model...')
# print label and f1-score for each
avg = 'weighted'
labels = Y.columns.tolist()
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
                      'accuracy': np.round(acc, 4)}, index=labels)


print('\n')
print('='*75)
print(f1_df)
print('\n')
print(f1_df.agg(['mean', 'median', 'std']))
print('='*75)
print('\n')

f1_df['f1-score'].mean()


#%%
print('\nCross-validating...\n')
scores = cross_val_score(
    pipeline,
    X_train.ravel(),
    y_train,
    scoring='f1_weighted',
    cv=3,
    n_jobs=N_JOBS)
print('\nCross-val scores:\n', scores)

#%%
# with open('results.txt', 'a') as file:
#     file.write('\n\n')
#     file.write(str(time.localtime()))
#     file.write(('-'*100))
#     file.write(str(pipeline.get_params()))
#     file.write('\n\n')
#     file.write(str(f1_df))
#     file.write('\n\n')
#     file.write(str(f1_df.agg(['mean', 'median', 'std'])))
#     file.write('\n\n')
#     file.write(('-'*100))
#     file.write('\n\n')


# ### 6. Improve your model
# Use grid search to find better parameters.

#%%
#import matplotlib.pyplot as plt
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(y_test[:, :].shape[1]):
#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
##%%
#plt.figure()
#lw = 2
#plt.plot(fpr[1], tpr[1], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
#
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()



# In[70]:

# GRID-SEARCH HYPERPARAMS

# print('Performing GridSearch. Please be patient ...')
# grid_params = {
#         'clf__estimator__C': [2, 4],
# #        'clf__estimator__n_estimators': [80, 120, 150],
# #        'clf__estimator__class_weight': [{0: 1, 1: 500},
# #                                         {0: 1, 1: 300},]

# }

# grid_cv = GridSearchCV(
#     pipeline,
#     grid_params,
#     cv=3,
#     scoring='f1_weighted',
#     n_jobs=2,
# )
# grid_cv.fit(X_train.ravel(), y_train)


# print('Using best params...')
# print(grid_cv.best_params_)

# y_pred = grid_cv.predict(X_test.ravel())

# print('Scoring model using tuned params...')
# # print label and f1-score for each
# avg = 'weighted'
# labels = Y.columns.tolist()
# f1 = []
# prec = []
# rec = []
# acc = []
# #train_scores = []
# for i in range(y_test[:, :].shape[1]):
#     f1.append(f1_score(y_test[:, i], y_pred[:, i], average=avg))
#     acc.append(accuracy_score(y_test[:, i], y_pred[:, i]))
#     rec.append(recall_score(y_test[:, i], y_pred[:, i], average=avg))
#     prec.append(precision_score(y_test[:, i], y_pred[:, i], average=avg))

# # summarize f1-scores and compare to the rate of positive class occurance
# f1_df = pd.DataFrame({'f1-score': np.round(f1, 4),
#                       'precision': np.round(prec, 4),
#                       'recall': np.round(rec, 4),
#                       'accuracy': np.round(acc, 4)}, index=labels)


# print('\n')
# print('='*75)
# print(f1_df)
# print('\n')
# print(f1_df.agg(['mean', 'median', 'std']))
# print('='*75)
# print('\n')


#%%
# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF
# * sampling: upsample or downsample in order to improve balance between classes
#  * however, upsampling minority classes may also affect majority classes and result in no significant imporvement
# * create more features
# * search each message for keywords; use target array labels as keywords
#  * for example, search for keywords, `food`, `water`, `shelter`...

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:





# # Resampling

# In[ ]:


# # review class balance
# print(y_train.sum() / y_train.shape[0])

# # combine train sets
# X_c = pd.concat([X_train, y_train], axis=1)

# fail = X_c[X_c['tools'] == 0]
# success = X_c[X_c['tools'] == 1]

# # upsample to match 'fail' class
# success_upsampled = resample(success,
#                              replace=True,
#                              n_samples=int(len(fail)*(0.25)),
#                              random_state=11
#                              )

# upsample = pd.concat([fail, success_upsampled])

# upsample['tools'].value_counts()

# upsample.columns.to_list()

# # split back into X_train, y_train
# X_train = upsample.drop('tools', axis=1)
# y_train = upsample['tools']

# print('X_train size:  ', X_train.shape[0])
# print('X_test size:   ', X_test.shape[0])
# print('X_holdout size:', X_holdout.shape[0])


# In[ ]:


# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight('balanced', np.unique(y), y)


# # Add more features

# In[ ]:




