# Disaster Response Pipeline

## Motivation
The goal of this project is to classify raw text disaster messages such that they can be forwarded automatically to appropriate relief agencies. This is something that is overwhelming for humans to classify when there may be thousands of messages coming in every minute. 

# File Description
```
 - **app/**: 
        |__ **run.py**: loads database file and model, renders plots, and classifies queries using loaded model
        |__ **templates/**
                |__ **go.html**: classification output page
                |__ **index.html**: homepage
 - **data/**: 
        |__ **process_data.py**: merges and cleans (2) CSV files, *disaster_messages.csv* and *disaster_categories.csv*, 
                                 then writes clean data to a database file
        |__ **disaster_messages.csv**: original data, contains messages
        |__ **disaster_categories.csv**: original data, contains categories
        |__ **disaster_response.db**: clean data
 - **models/**: 
        |__ **train_classifier.py**: loads from database, trains and tunes hyper-params, outputs a pickled model
        |__ **disaster_clf_.pkl**: pickled model
```

# Acknowledgments


# Structure
```
**disaster_response_project**
  |
  |___ **app**
  |        |
  |        |__ run.py
  |        |__ **templates**
  |                 |__ go.html
  |                 |__ index.html
  |
  |___ **data**
  |        |__ process_data.py
  |        |__ disaster_response.db
  |
  |___ **models**
            |__ train_classifier.py
            |__ disaster_clf.pkl
  ```
