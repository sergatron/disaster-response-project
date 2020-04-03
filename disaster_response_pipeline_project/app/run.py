import sys
sys.path.append('C:\\Users\\smouz\\OneDrive\\Desktop\\udacity\\ds_nanodegree\\data-engineering\\disaster-response\\disaster_response_pipeline_project\\models')

import re
import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from plotly.graph_objs import Bar
# from sklearn.externals import joblib

from joblib import dump, load
from sqlalchemy import create_engine

from flask import render_template, request, jsonify
from flask import Flask

app = Flask(__name__)


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

    return X, Y, df, category_names

#%%
# load data
# engine = create_engine('sqlite:///data/disaster_response.db')
# df = pd.read_sql_table('disaster_response', engine)

# load model
model = load("models/disaster_clf.pkl")

X, Y, df, category_names = load_data('data/disaster_response.db', 20000)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Renders graphs created from the loaded data.
    Renders homepage with visualizations of the data.

    """


    # extract data needed for visuals
    genre_counts = df['genre'].value_counts().values
    genre_names = df['genre'].value_counts().index.to_list()

    # extract category names and count of each
    category_names = list(Y.sum().sort_values(ascending=False).index)
    category_values = list(Y.sum().sort_values(ascending=False))

    # create visuals
    graphs = [
        {
            'data': [
                Bar(x = genre_names,
                    y = genre_counts)
                ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
                }
            },
        {
            'data': [
                Bar(x = category_names,
                    y = category_values)
                ],
            'layout': {
                'title': 'Target Category Count',
                'yaxis': {'title': 'Count',
                          'type': 'linear'
                          },
                'xaxis': {'title': 'Category',
                          'tickangle': -45,
                          }

                }

            },

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Renders a page which takes in user's query then passes
    the query to the model which makes predictions and outputs
    the labels to screen.
    """
    # save user input in query
    query = request.args.get('query', 'Invalid entry')

    toks = [tokenize(item) for item in [query]][0]

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()