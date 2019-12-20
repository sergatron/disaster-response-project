
import re
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from joblib import dump, load
from sqlalchemy import create_engine


import os
# os.chdir('../models')

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

    # extract directory name
    dir_ = re.findall(".*/", database_filepath)
    engine = create_engine(f'sqlite:///{database_filepath}')
    # extratc table name by stripping away directory name
    table_name = database_filepath.replace('.db', '').replace(dir_[0], "")

    df = pd.read_sql_table(f'{table_name}', engine)

    #           *** TEMPORARY SAMPLE TO TEST SCRIPT ***
    df = df.sample(4000)

    # explore `related` feature where its labeled as a `2`
    related_twos = df[df['related'] == 2]
    df.drop(index=related_twos.index, inplace=True)

    df = df.reset_index(drop=True)

    # define features and predictors
    X = df.loc[:, 'message']
    Y = df.loc[:, 'related':]


    # extract label names
    category_names = Y.columns.to_list()

    return X, Y, df, category_names

#%%
# load data
# engine = create_engine('sqlite:///data/disaster_response.db')
# df = pd.read_sql_table('disaster_response', engine)

# load model
model = load("models/disaster_clf.pkl")

X, Y, df, category_names = load_data('data/disaster_response.db')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():



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
    # save user input in query
    query = request.args.get('query', '')

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