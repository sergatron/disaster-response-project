#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[1]:


# import libraries
import os
import pandas as pd
import numpy as np

import sqlite3
from sqlalchemy import create_engine

#%%
pd.options.display.max_columns = 50

# In[2]:


print(os.getcwd())
print(os.listdir())


# In[3]:


# load messages dataset
messages_df = pd.read_csv('data/disaster_messages.csv')
messages_df.head()

if any(messages_df):
    print('success')


# In[4]:


messages_df.info()


# In[5]:


# load categories dataset
categories_df = pd.read_csv('data/disaster_categories.csv')
categories_df.head()


# In[6]:


categories_df.info()


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# In[7]:


# merge datasets
df = pd.merge(messages_df, categories_df, on='id')
df.head()


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.

# In[8]:


# create a dataframe of the 36 individual category columns
categories_df = categories_df['categories'].str.split(';', expand=True)
categories_df.head()


# In[9]:


# select the first row of the categories dataframe
# use `iloc` to select `row 0` and all columns (`:`)
row = categories_df.iloc[0, :]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything
# up to the second to last character of each string with slicing
category_colnames = row.str.replace("[-0-9]", '').tolist()
print(category_colnames)


# In[10]:


# rename the columns of `categories`
categories_df.columns = category_colnames
categories_df.head()


#%%
# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

# In[11]:


categories_df['related'].unique()


# In[12]:


# categories_df.loc[:, 'related'].apply(lambda x: x[-1])
# categories_df.loc[:, column].astype('str').str.replace("[a-z-]", '')


# In[13]:


for column in categories_df:
    # set each value to be the last character of the string
    categories_df[column] = categories_df.loc[:, column].astype('str').apply(lambda x: x[-1])

    # convert column from string to numeric
    categories_df[column] = categories_df[column].astype('int')
categories_df.head()



#%%
# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.

# In[14]:

# drop column 'original', as it contains too many missing values
df.drop('original', axis=1, inplace=True)

# drop the original categories column from `df`
df.drop('categories', axis=1, inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories_df], axis=1)

df.info()


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# df.duplicated??
# df.drop_duplicates??


# In[16]:

# check number of duplicates
df[df.duplicated(subset='id', keep=False)].shape[0]

messages_df[messages_df.duplicated(subset='id', keep=False)]


# In[17]:

# drop duplicates
df.drop_duplicates(subset = 'id', inplace = True)


# In[18]:


# check number of duplicates
df[df.duplicated(subset='id', keep=False)].shape[0]

# print warning if any duplicates are found
if df[df.duplicated(subset='id', keep=False)].any().any():
    print('WARNING: Contains duplicate rows!')

# ## Missing Values

# In[24]:


df.genre.value_counts()


# In[20]:


# Missing values
df.isnull().sum()


# In[22]:


df[df.related.isnull()]


# In[19]:


# amount of missing values is relatively small
# drop rows with missing values
df.dropna(subset=['related'], inplace=True)


# In[20]:


# check for missing values again
df.isnull().sum().any()

if df.isnull().sum().any():
    print('WARNING: Contains missing values!')

# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

# In[21]:


df.info()


# In[22]:


conn = sqlite3.connect('message_cat.db')

# get a cursor
cur = conn.cursor()

# drop table if already exists
cur.execute("DROP TABLE IF EXISTS message_cat")
conn.commit()
conn.close()


# In[23]:


engine = create_engine('sqlite:///message_cat.db')
df.to_sql('message_cat', engine, index=False)


# In[24]:


ls


# In[25]:


# test connection to database
conn = sqlite3.connect('message_cat.db')

# get a cursor
cur = conn.cursor()

# create the test table including project_id as a primary key
df = pd.read_sql("SELECT * FROM message_cat", con=conn)
print(df.head())
conn.commit()
conn.close()


# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[ ]:


