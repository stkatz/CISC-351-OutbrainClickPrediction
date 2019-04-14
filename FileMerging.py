"""
Project: Outbrain Click Prediction
Task: Predict which recommended content each user will click.
Group: 5
Group members: Analeeze Mendonsa, Samathan Katz, and Rachel Venis
Date: April 3rd, 2019 

Document Description: This Files Manipulates the Kaggle Documents given to create 
data files that we can work with.
"""
# %% Run Modules 
import matplotlib.pyplot as plt #MatPlotLib: Module for data visualization
import matplotlib as mlp
import pandas as pd #Pandas: Module used for data manipulation and analysis
import numpy as np #Numpy: Module for scientific computing 
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# %% Reads Clicks_train CSV Files
train = pd.read_csv('../input/clicks_train.csv')

# %% Reads Events CSV Files
events = pd.read_csv('../input/events.csv')

# %% Merges train file and US column in events file together to create a new file called joined_events_train
join_data = pd.merge(train, events_us, on='display_id', how='inner')
join_data.to_csv('../input/joined_events_train.csv')

# %% Reads Joined_data CSV Files
joined_data = pd.read_csv('../input/joined_events_train.csv')

# %% Drops all other countries except for US as our project. It creates a new csv file with only US events
us_events = events.loc[events['geo_location'].str.split('>').str[0]=='US']
us_events.to_csv('../input/events_us.csv')


# %% Reads Events_US CSV Files
events_us = pd.read_csv('../input/events_us.csv')

# %% We separate the data by "US"
joined_data['Country'] = joined_data['geo_location'].str[0:2]
joined_data.head(10)

# %% We separate the data by the "state"
joined_data['State'] = joined_data['geo_location'].str[3:5]
joined_data.head(10)

# %% Convert the file earlier into a new CSV Files
joined_data.to_csv('../input/new_join.csv')

# %% Open the New_join CSV file as Join_final
join_final = pd.read_csv('../input/new_join.csv')

# %% Sorts the join_final in unique values 
join_final['State'].unique()

# %% 
join_final = join_final[join_final['Country'] != '--']
join_final = join_final[join_final['Country'] != 'A2']
join_final = join_final[join_final['State'] != 'AA']


# %% Drops the GeoLocation in the join_final file
join_final.drop(['geo_location'], axis=1 , inplace=True)
join_final.head(10)

# %% Drops the Unnamed in the join_final file
join_final.drop(['Unnamed: 0.1.1'], axis=1,  inplace=True)
join_final.drop(['Unnamed: 0.1.1.1'], axis=1,  inplace=True)
join_final.head(10)

# %% Converts all the join_final into a new final_train file
join_final.to_csv('../input/final_train.csv')
# %% Reads Final_train CSV file as train
train = pd.read_csv('../input/final_train.csv', index_col=0)

# %%  Reads Document_category file 
document_category = pd.read_csv('../input/documents_categories.csv')

# %% Merges Train File and the average_category_confidenceLevel column from document_categories to create a new variable train_events_cat
average_category_confidenceLevel = pd.DataFrame(document_category.groupby(['document_id'])['confidence_level'].mean()).reset_index()
train_events_cat = pd.merge(train, average_category_confidenceLevel, how='left',on='document_id')
train_events_cat = train_events_cat.dropna()

# %% Creates a copy of train_events_cat
train = train_events_cat.copy()

# %% Drops the Country Column from file
train.drop(['Country'],axis=1,inplace=True)
train = train.dropna()

# %% Creates a copy of train file and drops the State file.
integer_encoded = train.copy()
label_encoder = LabelEncoder()
integer_encoded['State_Code'] = label_encoder.fit_transform(integer_encoded['State'])
integer_encoded.drop(['State'], axis=1, inplace=True)