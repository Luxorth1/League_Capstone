#!/usr/bin/env python
# coding: utf-8

# # League Of Legends Capp.GG
# 
# This project is focused on providing post-game analysis on League of Legends games.
# 
# This will include Machine Learning driven outcome predictions along with multiple other metrics that can be utilized to improve gameplay.

# ### Data Gathering
# 
# We will be utilizing the RIOT API to gather player data, as well as match statistics.
# 
# The Riotwatcher 3.1.1 library will be used for all API calls. Our data will consist of player matches from Diamond I rank which can be regarded as 'higher' level of play. 

# In[1]:


from riotwatcher import LolWatcher, ApiError
import pandas as pd
import json
import time
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, plot_confusion_matrix

from sklearn.metrics import make_scorer, precision_recall_curve

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
random_state = 343


# In[2]:


# global API variables
api_key = 'RGAPI-18b1b20f-deb8-4ada-8b0e-49e766747747'
watcher = LolWatcher(api_key)
my_region = 'na1'


# In[69]:


# data collection functions
def list_of_dict_data_to_df(data):
    '''
    Takes a list of dictionaries and creates a dataframe
    
    ARGS:
        data: list of dictionaries
            TYPE: List
    returns DataFrame from list of dictionaries
    '''
    df = pd.DataFrame()
    for idx, d in enumerate(data):
        ph_df = pd.DataFrame.from_dict(data[idx], orient='index')
        ph_df = ph_df.T
        df = pd.concat([df, ph_df], sort = False)
    return df
def get_summonerId_list(df):
    '''
    Takes in dataframe with summonerId column and returns summonerId list
    ARGS:
        df: dataframe with SummonerId Column
            TYPE: DataFrame
    returns list of SummonerIds from dataframe
    '''
    summoner_list = df['summonerId']
    return summoner_list
def get_acc_id_from_summ_id(summonerId_list, lag=3):
    '''
    Calls RIOT API for accountId from SummonerIds
    ARGS:
        summonerId_list: list of summonerIDs
            TYPE: list
        lag: time to wait for API call (Default: 3)
            TYPE: int
    returns list of accountIds
    '''
    acc_list = []
    for summoner in summonerId_list:
        req = watcher.summoner.by_id(my_region, summoner)
        time.sleep(lag)
        acc_list.append(req['accountId'])
    return acc_list
def get_gameId_from_accId(accountId_list, lag = 3):
    '''
    Calls RIOT API for gameIds from AccountIds
    ARGS:
        accountId_list: list of accountIds
            TYPE: list
        lag: time to wait for API call (Default: 3)
            TYPE: int
    returns list of gameIds
    '''
    matchlist = []
    for idx, account in enumerate(accountId_list):
        req = watcher.match.matchlist_by_account(my_region, account)
        time.sleep(3)
        for index, match in enumerate(req['matches']):
            matchlist.append(req['matches'][index]['gameId'])
    return(matchlist)
def team_data_from_match(gameId_list, lag = 3):
    '''
    Calls RIOT API for match data and extracts team data from AccountIds
    ARGS:
        gameId_list: list of gameIds
            TYPE: list
        lag: time to wait for API call (Default: 3)
            TYPE: int
    returns DataFrame of overall team data from a match
    '''
    team_data = pd.DataFrame()
    start = time.time()
    list_of_game_id_list = [gameId_list[x:x+100] for x in range(0, len(gameId_list), 100)]
    for game_id_list in list_of_game_id_list:
        for game_data in game_id_list:   
            try:
                req = watcher.match.by_id(my_region, game_data)
                print('Done')
            except: 
                end = time.time()
                print(end - start)
            time.sleep(1.5)
            for team in req['teams']:
                try:
                    ph_team_df = pd.DataFrame.from_dict(team)
                    team_data = pd.concat([team_data, ph_team_df], sort = False)
                except:
                    print('No data recieved from {}'.format(game_data))
    end = time.time()
    print(end-start)
    return team_data
# preprocessing functions and single data grab
def call_games(summonerName):
    '''
    Calls RIOT API for team data from a single summoner's name
    ARGS:
        summonerName: Name of summoner we need information on
            TYPE: String
    returns DataFrame of last 10 games' team data
    '''
    req = watcher.summoner.by_name(my_region, summonerName)
    account_list = []
    account_list.append(req['accountId'])
    matches = get_gameId_from_accId(account_list, 1)
    team_data = team_data_from_match(matches[:9], 1)
    return(team_data)
def clean_data(team_data):
    team_data = team_data.iloc[::5, :]
    team_data.reset_index(inplace=True)
    for col in team_data:
        if col == 'vilemawKills' or col == 'dominionVictoryScore' or col == 'bans':
            team_data.drop(columns = col, inplace = True)
    team_data['win'] = team_data['win'].map(lambda x: 1 if x == 'Win' else 0)
    return(team_data)
def process_data(team_data):
    y = team_data['win']
    x = team_data.loc[:, team_data.columns != 'win']
    x['teamId'] = x['teamId'].map(lambda x: 'Red' if x == 100 else 'Blue')
    return(x,y)
def scale_data(team_data, scaler):
    for col in team_data.columns:
        if col == 'index':
            team_data.drop(columns = col, inplace = True)
    X_num = team_data.select_dtypes(include = np.number)
    X_cat = team_data.select_dtypes(include = [object,bool])
    for column in X_cat.columns:
        if column != 'teamId':
            X_cat[column] = X_cat[column].map(lambda x: 1 if x == True else 0)
    #fit and transform numerical data with scaler
    X_num_scaled = scaler.transform(X_num)
    #recombine as DF
    X_num_scaled_df = pd.DataFrame(X_num_scaled, columns = X_num.columns)
    X_num_scaled_df.reset_index()
    X_cat.reset_index()
    team_data = pd.concat([X_num_scaled_df, X_cat], axis = 1)
    team_data['teamId'] = team_data['teamId'].map(lambda x: 1 if x == 'Red' else 0)
    return(team_data)
def full_process(summonerName, scaler):
    team_data = call_games(summonerName)
    team_data_clean = clean_data(team_data)
    team_data_x, team_data_y = process_data(team_data_clean)
    team_data_x = scale_data(team_data_x, scaler)
    return(team_data_x, team_data_y)
def validation_process(team_data, scaler):
    team_data_clean = clean_data(team_data)
    team_data_x, team_data_y = process_data(team_data_clean)
    team_data_x = scale_data(team_data_x, scaler)
#     for col in team_data_x.columns:
#         if col == 'index':
#             team_data_x.drop(columns = col, inplace = True)
    return(team_data_x, team_data_y)


# In[4]:


# # begin to call data from RIOT API

# diamond_data = watcher.league.entries(my_region, 'RANKED_SOLO_5x5', 'DIAMOND', 'I')
# # process first dictionary
# league_data_df = list_of_dict_data_to_df(diamond_data)
# summonerId_list = get_summonerId_list(league_data_df)
# accountId_list = get_acc_id_from_summ_id(summonerId_list)
# gameId_list = get_gameId_from_accId(accountId_list)
# # short_game = gameId_list[:-20450]
# team_data = team_data_from_match(gameId_list)


# In[5]:


# team_data.head()


# In[6]:


# team_data.to_csv(r'master_data.csv')


# ## Data Exploration
# 
# Now that we have completed our API calls, we will begin to clean and explore our data. Our data contains fields from different game modes, however we are only taking a look at 'Summoner's Rift' the main 5v5 game mode. Therefore, we will get rid of the fields that apply to other game modes outside of Summoner's Rift.

# In[7]:


team_data = pd.read_csv('master_data.csv')


# In[8]:


team_data.drop(columns=['vilemawKills', 'dominionVictoryScore', 'Unnamed: 0','bans'], inplace = True)


# In[9]:


team_data.head()


# In[10]:


team_data.isna().all()


# In[11]:


for column in team_data:
    print(team_data[column].unique())


# Next we are going to break out our team color designation to see if there are any trends with what team wins more in Diamond I. According to the documentation for the Riot API we see that 100 - Blue Team and 200 - Red Team, the teams don't matter too much, only determine which side of the map your team starts on. 

# In[12]:


# df['Event'].mask(df['Event'] == 'Hip-Hop', 'Jazz', inplace=True)
team_data['teamId'] = team_data['teamId'].apply(lambda x: 'Blue' if x == 100 else 'Red')
team_data['win'] = team_data['win'].apply(lambda x: 'Loss' if x == 'Fail' else 'Win')


# In[13]:


team_data


# In[14]:


team_data.head()


# Now that we have all of our data in one area, we are going to resample our data since the way the data is organized doesn't quite fit into what we are going for. Our resampling will take every 5 rows and add that to a new dataframe, then we will save it to a csv as our resampled master data.

# In[15]:


full_resampled_team_data = team_data.iloc[::5, :]


# In[16]:


full_resampled_team_data.head()


# In[17]:


full_resampled_team_data.to_csv(r'resampled_master_data.csv')

full_resampled_team_data = pd.read_csv('resampled_master_data.csv')

full_resampled_team_data.drop(columns = 'Unnamed: 0', inplace = True)


# We will continue exploring our data, we will also move into the feature engineering portion to begin creating a model for our data.

# In[18]:


full_resampled_team_data.info()


# ### Train-Test Split
# 
# We will be splitting our original resampled data into a 75/25 ratio of train set and test set respectively. This is to prevent any data snooping bias while processing our training data. Once we finish processing and training our models on the training set we will run our test data through the processing pipeline and run it through the determined 'best' model to validate our model further.

# In[19]:


train_set, test_set = train_test_split(full_resampled_team_data, test_size = .25, stratify= full_resampled_team_data['win'], random_state=343)
print(train_set.shape)
print(test_set.shape)


# In[20]:


resample_team_data = train_set.reset_index(drop=True).copy()


# It does not seem like we need to do any data type changes, the data is generally clean. I would like to take a look and see if there are any extreme cases of multicolinearity we will create a correlation heat map to determine this.

# In[21]:


plt.figure(figsize= (12,12))

corr = resample_team_data.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, center = 0, vmin = -1, vmax = 1, square=True, linewidths=.5, annot = True, fmt='.2f' ,cmap=sns.diverging_palette(10,220,sep = 80, n = 100))


# We can see that there is a few entries with decently high correlation values. However, the nature of these features compared to our subject matter knowledge leads us to accept the risk of multicolinearity and keep all of these features. Now we are going to take a look at our categorical data and find out if there are any inconsistencies.

# In[22]:


categoricals = resample_team_data.select_dtypes(include=bool).columns

for col in categoricals:
    print("Feature: {}\n{} \n".format(col, resample_team_data[col].value_counts(normalize=True)))


# I would like to take a look and see if we can notice any trends with our boolean data. Does getting an Inhibitor first lead to more wins? Does the Blue team have a higher chance of getting the first Rift Herald or Dragon?

# In[23]:


sns.countplot(x = resample_team_data['firstBaron'], hue = resample_team_data['win'])


# Based on this graph above we can see something very interesting. It would seem that among wins, it looks like a 50/50 split between getting the first Baron kill or not. Whereas, among our losses it seems that if you kill Baron first you are much less likely to lose the game.

# In[24]:


resample_team_data['y_num'] = resample_team_data['win'].apply(lambda x: 1 if x == 'Win' else 0)
dfTemp = resample_team_data[(resample_team_data['y_num'] == 1)]
fig = px.treemap(dfTemp,
                path = ['teamId','dragonKills'],
                values= 'y_num', title='Teams and Dragon Kills')
fig.show()


# We can see that this Tree Map seems to show us that the Red team tends to kill more dragons vs. the Blue team. Lets explore this further, I would like to see the team color, wins, and dragon kills.

# In[25]:


sns.countplot(x = resample_team_data['dragonKills'], hue = resample_team_data['win'])


# So here, we can pretty clearly see that there is pretty evident data to show that more dragons leads to more wins, but of the teams that won, how many of them were on each team that also more dragons?

# In[26]:


# create df with only wins
tempDF = resample_team_data.loc[(resample_team_data['win'] == 'Win')]
sns.countplot(x = resample_team_data['dragonKills'], hue = resample_team_data['teamId'])
plt.legend(loc='upper right')


# So, out of all the winning games we see that the Red team tends to do better getting a higher number of dragon kills in games they won.
# 
# ### Feature Engineering
# 
# Next we will move into preprocessing our data so we can begin using modeling techniques to test the effectiveness of the model. 
# 
# Our first step will be to break out our target feature from our other features. We will also reset our index before splitting our data to maintain index integrity. We will also code our target feature to numeric values 0 being a loss and 1 being a win.
# 

# In[27]:


df = resample_team_data.reset_index(drop=True)
X = df.drop(columns='win', axis = 1)
y = df['win'].copy()


# In[28]:


y = y.map(lambda x: 1 if x == 'Win' else 0)


# In[29]:


df.head()


# In[30]:


X.drop(columns='y_num', axis =1, inplace= True)


# ##### Feature Scaling
# 
# Here we are going to use a StandardScaler to normalize our numerical values.

# In[31]:


# classify our numerical data
X_num = X.select_dtypes(include=np.number)
# classify categorical data
X_cat = X.select_dtypes(include=[object,bool])
# create scaler instance
MMscaler = MinMaxScaler().fit(X_num)
print(X_num.columns)
#fit and transform numerical data with scaler
X_num_scaled = MMscaler.transform(X_num)

#recombine as DF
X_num_scaled_df = pd.DataFrame(X_num_scaled, columns = X_num.columns)
X_num_scaled_df.head()


# In[32]:


# team_data_x = scale_data(team_data_x, MMscaler)
lux_x, lux_y = full_process('Luxorth', MMscaler)


# In[33]:


lux_x.head()


# Lets combine our scaled data with our original categorical data.

# In[34]:


X = pd.concat([X_num_scaled_df, X_cat], axis = 1)
X.head()


# ##### Categorical Feature Encoding
# Here we will encode our boolean data and categorical data into 0s and 1s. The reason we are not using One-Hot Encoding is to avoid multicolinearity, as well and the curse of dimensionality. If we can encode these in a single feature and encapsulate the translation of the data without adding additional fields it will provide more optimal model results.

# In[35]:


X['teamId'] = X['teamId'].map(lambda x: 1 if x == 'Red' else 0)
for column in X_cat.columns:
    if column != 'teamId':
        X[column] = X[column].map(lambda x: 1 if x == True else 0)


# In[36]:


X.head()


# Now that all of our data is in a state that we can pump into a machine learning algorithm we will move into creating a pre-processing pipeline that has the ability to take an arbitrary Summoner Name, and pump out a list of the last 10 games the player participated in, clean the data and prepare it for predictions.

# ### Modelling
# 
# We will being utilizing our engineered data to create our predictive models.
# 
# ##### Create Evaluation Test Set
# 
# Here we will create another test set from our training set that we will pump into our model to evaluate performance.

# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = random_state)


# In[38]:


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# ##### Choice of Evaluation Metric
# 
# We can recall that our target class is quite balanced, we will reproduce this observation below.

# In[39]:


plt.figure(figsize=(10,10))
sns.countplot(resample_team_data['win'])
plt.xlabel('Wins vs Loss')
plt.title('Distribution of Match Outcomes')
plt.show()


# In[40]:


resample_team_data['win'].value_counts()


# Let's explore the various metrics we have available to determine how 'well' our model performs.
# 
# We can begin with a simple confusion matrix which shows all possible outcomes. 
# 
# The goal in this project is to predict whether a team 'should' have won a game, so a high Accuracy would be optimal in this situation. Therefore, we will want to optimize the F1 Threshold of this model.
# 
# These TP/FP/TN/FN values can be used to produce a few metrics that give us a better idea of how our model is performing. We will now explore Accuracy, Precision, and Recall below:
# 
# ###### Accuracy:
# Accuracy simply defines the number of correct predictions over the number of observations.
# 
# Accuracy = (TN + TP) / # of Observations
# 
# We would prefer for our model to predict based on the features alone, this predictive model is geared more towards letting a player know whether they 'should' have won based on how the team played with objectives.
# 
# ###### Precision: 
# Precision determine the amount of true predictions. 
# Precision = TP / (TP + FP). 
# 
# This calculation would be fantastic if we wanted to minimize false positives, however in our analysis it is more important to minimize false negatives for our situation. The precision score will not be the metric we focus on.
# 
# ###### Recall: 
# Recall defineds out of all true observations, how many were accurately predicted?
# 
# Recall = TP / (TP+FN)
# 
# This metric would be chosen if the goal was to predict accurately on every single match, however since we are giving an objective opinion based on data we will want to maximize our overall score rather than reducing a specific TP/TN/FP/FN value.
# 
# ###### ROC and AUC
# Receiver Operating Characteristics (ROC) is developed by plotting the True Positive Rate, and Associated Area Under Curve (AUC). We will use the Youden's J statistic to determine the optimal point on the curve.
# 
# ###### F1 Threshold
# This metric shows us the weighted average between recall and precision. This tends to be the best way to determine a model's performance, we would like to maximize this value as it is a weighted average between recall and precision, it will allow us wiggle room with situations where a team won, but the data suggests they maybe should not have won a game.
# 
# ### Baseline Model
# 
# We will begin with a baseline Logistic Regression model and base future model performance on this baseline.

# In[41]:


#instantiate logreg
logreg = LogisticRegression(fit_intercept = True, C=1e17, random_state = random_state)

#fit model
logreg.fit(X_train, y_train)

#predictions
train_preds = logreg.predict(X_train)
test_preds = logreg.predict(X_test)

print('Classification Report for Resampled Train Set')
print(classification_report(y_train, train_preds))
print('Classification Report for Test Set')
print(classification_report(y_test, test_preds))
#product feature importance

importance = logreg.coef_

# summarize feature importance
for i, v in enumerate(importance[0]):
    print('Feature {}, Score: {}'.format(X_train.columns[i],v))
plt.bar([x for x in range(len(importance[0]))], importance[0])
plt.show()


# In[42]:


def Conf_Plt(estimator, X_train, y_train, X_test, y_test):
    '''
    Plots confusion matrix for both train and test sets
    
    ARGS:
        estimator: Instance of estimator
        X_train: training features
        y_train: training target
        X_test: test features
        y_test: test target
    returns None
    '''
    fig,axes = plt.subplots(1,2, figsize = (10,6), sharey = 'row')
    plot_confusion_matrix(estimator, X_train, y_train, values_format = '.0f', ax = axes[0])
    axes[0].set_title('Training Set Confusion Matrix')
    plot_confusion_matrix(estimator, X_test, y_test, values_format = '.0f', ax = axes[1])
    axes[1].set_title('Test Set Confusion Matrix')
    plt.show()
    return None


# In[43]:


Conf_Plt(logreg, X_train, y_train, X_test, y_test)


# Now that we have a baseline model, I would like to take a look at the format of our data a bit more. We can see that we have a number of features that describe who got an objective first, and then we have game totals, I would like to explore what our model results would look like if we separated these. Does total scores represent the data better than who got objectives first? Let's find out.

# In[44]:


bool_feat = ['teamId', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald']
num_feat = ['towerKills', 'inhibitorKills', 'baronKills', 'dragonKills', 'riftHeraldKills']

bool_feat_train_x = X_train[bool_feat]
bool_feat_test_x = X_test[bool_feat]
bool_feat_train_y = y_train
bool_feat_test_y = y_test
num_feat_train_x = X_train[num_feat]
num_feat_test_x = X_test[num_feat]
num_feat_train_y = y_train
num_feat_test_y = y_test


# In[45]:


#instantiate logreg
logreg_bool = LogisticRegression(fit_intercept = True, C=1e17, random_state = random_state)

#fit model
logreg_bool.fit(bool_feat_train_x, bool_feat_train_y)

#predictions
train_preds = logreg_bool.predict(bool_feat_train_x)
test_preds = logreg_bool.predict(bool_feat_test_x)

print('Classification Report for Resampled Train Set')
print(classification_report(bool_feat_train_y, train_preds))
print('Classification Report for Test Set')
print(classification_report(bool_feat_test_y, test_preds))
#product feature importance

importance = logreg_bool.coef_

# summarize feature importance
for i, v in enumerate(importance[0]):
    print('Feature {}, Score: {}'.format(X_train.columns[i],v))
plt.bar([x for x in range(len(importance[0]))], importance[0])
plt.show()


# In[46]:


#instantiate logreg
logreg_num = LogisticRegression(fit_intercept = True, C=1e17, random_state = random_state)

#fit model
logreg_num.fit(num_feat_train_x, num_feat_train_y)

#predictions
train_preds = logreg_num.predict(num_feat_train_x)
test_preds = logreg_num.predict(num_feat_test_x)

print('Classification Report for Resampled Train Set')
print(classification_report(num_feat_train_y, train_preds))
print('Classification Report for Test Set')
print(classification_report(num_feat_test_y, test_preds))
#product feature importance

importance = logreg_num.coef_

# summarize feature importance
for i, v in enumerate(importance[0]):
    print('Feature {}, Score: {}'.format(X_train.columns[i],v))
plt.bar([x for x in range(len(importance[0]))], importance[0])
plt.show()


# #### Classifiers
# 
# We see that we are still obtaining information from our features even separately. We will continue with our full data set and move towards finding the optimal classifier to model our data successfully. Once we have a better idea of what classifiers to use, we will then begin to tune them to find the optimal model.

# In[47]:


# create classifier dictionary

clf_dict = {
    'Logistic Regression': LogisticRegression(random_state=random_state),
    'Decision Tree': DecisionTreeClassifier(random_state=random_state),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Classification': SVC(random_state=random_state, probability= True),
    'Random Forest': RandomForestClassifier(random_state = random_state),
    'Adaboost': AdaBoostClassifier(random_state=random_state),
    'Gradient Boosting': GradientBoostingClassifier(random_state = random_state),
    'XGBoost': xgb.XGBClassifier()
}


# In[48]:


def batch_clf(X_train, y_train, X_test, y_test, clf_dict, verbose = False):
    '''
    This function fits a dict of classifiers, makes predictions, plots ROC, returns metrics
    
    args:
        X_train: {array-like, sparse matrix} of shape (n_samples, n_features) train input features
        y_train: array-like of shape (n_samples) train target values
        X_test: {array-like, sparse matrix} of shape (m_samples, m_features) test input features
        y_test: array-like of shape (m_samples) test target values
        clf_dict: dictionary, key name is classifier name, and value is classifier instance
        verbose: if True, prints time taken to fit and predict for each classifier
    '''
    
    times = []
    train_acc_scores = []
    test_acc_scores = []
    train_f1_scores = []
    test_f1_scores = []
    train_precision_scores = []
    test_precision_scores = []
    train_recall_scores = []
    test_recall_scores = []
    train_roc_data = []
    test_roc_data = []
    
    # loop through clf_dict
    for key, clf in clf_dict.items():
        start_time = time.time()
        # fit
        clf_fit = clf.fit(X_train, y_train)
        
        # predict
        train_preds = clf_fit.predict(X_train)
        test_preds = clf_fit.predict(X_test)
        
        #accuracy scores
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        train_acc_scores.append(round(train_acc,2))
        test_acc_scores.append(round(test_acc,2))
        
        # F1 Score
        train_f1 = f1_score(y_train, train_preds)
        test_f1 = f1_score(y_test, test_preds)
        train_f1_scores.append(round(train_f1,2))
        test_f1_scores.append(round(test_f1, 2))
        
        # precision score
        train_precision = precision_score(y_train, train_preds)
        test_precision = precision_score(y_test, test_preds)
        train_precision_scores.append(round(train_precision, 2))
        test_precision_scores.append(round(test_precision,2))
        
        # recall scores
        train_recall = recall_score(y_train, train_preds)
        test_recall = recall_score(y_test, test_preds)
        train_recall_scores.append(round(train_recall, 2))
        test_recall_scores.append(round(test_recall,2))
        
        # probability preds
        train_hat = clf_fit.predict_proba(X_train)
        train_proba = train_hat[:,1]
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_proba)
        train_roc_data.append([fpr_train, tpr_train, thresholds_train])
        
        test_hat = clf_fit.predict_proba(X_test)
        test_proba = test_hat[:,1]
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, test_proba)
        test_roc_data.append([fpr_test, tpr_test, thresholds_test])
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        times.append(round(time_elapsed,2))
        
        if verbose:
            print('trained {} in {}'.format(key, round(time_elapsed,2)))
    # create df from results
    results = pd.DataFrame({'Model': list(clf_dict.keys()),
                           'Time': time,
                           'Train Acc': train_acc_scores,
                           'Test Acc': test_acc_scores,
                           'Train F1': train_f1_scores,
                           'Test F1': test_f1_scores,
                           'Train Precision': train_precision_scores,
                           'Test Precision': test_precision_scores,
                           'Train Recall': train_recall_scores,
                           'Test Recall': test_recall_scores,
                           })
    fig, axes = plt.subplots(1,2, figsize=(15,10))
    
    for i in range(len(train_roc_data)):
        axes[0].plot(train_roc_data[i][0], train_roc_data[i][1], lw = 4, 
                     label = '{}'.format(list(clf_dict.keys())[i]))
    for i in range(len(test_roc_data)):
        axes[1].plot(test_roc_data[i][0], test_roc_data[i][1], lw=4,
                    label = '{}'.format(list(clf_dict.keys())[i]))
    for ax in axes:
        ax.plot([0,1], [0,1], color='blue', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='upper right')
    axes[0].set_title('ROC curve - Train Set')
    axes[1].set_title('ROC curve - Test Set')
    plt.show()
    return results


# In[49]:


outcomes = batch_clf(X_train, y_train, X_test, y_test, clf_dict, verbose=True)
outcomes


# Based on the results seen above we see that Support Vector Classification performed quite well. We will now explore the same models using Randomized Search Cross Validation.

# In[50]:


RF_params = {'clf__max_depth': [4,5,6,10,15],
             'clf__criterion': ['gini', 'entropy'],
             'clf__n_estimators': [100, 200, 300],
             'clf__min_samples_split': [3,5,10],
             'clf__min_samples_leaf': [3,4,6]
            }
DT_params = {'clf__max_depth': [5,25,50],
             'clf__criterion': ['gini', 'entropy'],
             'clf__min_samples_split': [3,5,10]
            }
LR_params = {'clf__solver': ['liblinear'],
             'clf__C':[0.1,1]
            }
KNN_params ={'clf__n_neighbors': [3,5,11,15],
             'clf__weights': ['uniform', 'distance'],
             'clf_p': [1,5]
            }
AB_params = {'clf__n_estimators': [20,50,100,200],
             'clf__learning_rate': [0.5,1,1.5]}
GB_params = {'clf__n_estimators': [20,50,100,200],
             'clf__max_depth': [3,5,10,15]}
XGB_params ={'clf__n_estimators': [20,50,100,200], 
             'clf__max_depth': [3,5,10,15],
             'clf__min_child_weight': [2,3,5]}


# In[51]:


# create new dictionary
tuning_clf_dict = {}

for k,v in clf_dict.items():
    tuning_clf_dict[k] = {}
    # add clfs
    tuning_clf_dict[k]['clf'] = v
#remove SVM from clf
del tuning_clf_dict['Support Vector Classification']
del tuning_clf_dict['K-Nearest Neighbors']
#Add pipeline to nested dictionary
for k in tuning_clf_dict.keys():
    tuning_clf_dict[k]['pipeline'] = Pipeline([('clf', tuning_clf_dict[k]['clf'])])

# add key param_grid and default empty dict val
for k in tuning_clf_dict.keys():
    tuning_clf_dict[k]['param_grid'] = {}
    
#reassign param grid to param classifier
tuning_clf_dict['Logistic Regression']['param_grid'] = LR_params
tuning_clf_dict['Decision Tree']['param_grid'] = DT_params
tuning_clf_dict['Random Forest']['param_grid'] = RF_params
tuning_clf_dict['Adaboost']['param_grid'] = AB_params
tuning_clf_dict['Gradient Boosting']['param_grid'] = GB_params
tuning_clf_dict['XGBoost']['param_grid'] = XGB_params

# add RandomizedSearchCV to nested dict
cv = 3
scoring = 'recall'

for k in tuning_clf_dict.keys():
    tuning_clf_dict[k]['rscv'] = RandomizedSearchCV(estimator = tuning_clf_dict[k]['pipeline'],
                                                    param_distributions = tuning_clf_dict[k]['param_grid'],
                                                    scoring = scoring,
                                                    cv = cv)
# create new dict with clf names and RSCV object
# allows to be passed into batch_clf function
rscv_dict = {}

for k in tuning_clf_dict.keys():
    rscv_dict[k] = tuning_clf_dict[k]['rscv']


# In[52]:


rscv_outcome = batch_clf(X_train, y_train, X_test, y_test, rscv_dict, verbose = True)
rscv_outcome


# We see that with Randomized Search Cross Validation it really only increased our performance marginally. Also, it drastically increased execution time for some of our models. We will go ahead and stick with our Decision Tree as we can continue to tune this model and ensure that it is as optimal as possible.

# In[53]:


# create Decision Tree Instance

decision_tree = DecisionTreeClassifier(random_state=random_state)

# define pipeline
pipeline = Pipeline([('clf', decision_tree)])

# define paramater grid
DT_params = {'clf__max_depth': [5,25,50],
             'clf__criterion': ['gini', 'entropy'],
             'clf__min_samples_split': [3,5,10]
            }
cv = 5

# define GridSearch
decision_tree_grid = GridSearchCV(estimator = pipeline, param_grid = DT_params, scoring = 'f1', cv=cv)

# fit gridsearch

fitted_DT_grid = decision_tree_grid.fit(X_train, y_train)

# get preds
train_pred = fitted_DT_grid.predict(X_train)
test_pred = fitted_DT_grid.predict(X_test)

# get probabilities
train_hat = fitted_DT_grid.predict_proba(X_train)
train_probs = train_hat[:,1]
test_hat = fitted_DT_grid.predict_proba(X_test)
test_probs = test_hat[:,1]


# In[54]:


print(classification_report(y_test, test_pred))


# Now that we have a well performing model, we need to dig deeper into what features are more important when it comes to this model. This will allow us to provide the intended recommendations to the player in order to provide data-driven gameplay tips. 

# In[55]:


model_best_param = fitted_DT_grid.best_params_
model_best_param


# In[56]:


feature_importances = fitted_DT_grid.best_estimator_[0].feature_importances_


# In[57]:


for i, f in enumerate(feature_importances):
    print('Feature: {}'.format(X_train.columns[i]))
    print('Importance: {}'.format(f))


# We can see that Inhibitor Kills tend to be of greater importance in the grand scheme. Shooting for an early first Tower also makes a large impact. However, lets go ahead and get these values on a graph to have a better visual understanding of what we have here.

# In[58]:


features = pd.Series(fitted_DT_grid.best_estimator_[0].feature_importances_, index = X_train.columns)
# sort series
features = features.sort_values(ascending = True)
#Drop our 0 values
features = features[features !=0]
# plot graph
features.plot(kind = 'barh', figsize=(15,20))
plt.title('Significant Features')
plt.show()


# Now, we are going to go ahead and pump in our original test/validation set to give the model new, untouched data to work with. We will also create a full pipeline that we can pump fresh data into.

# In[70]:


val_set_cleaned = clean_data(test_set)
val_set_cleaned_processed_x, val_set_cleaned_processed_y = process_data(test_set)


# In[71]:


val_set_cleaned_processed_x.head()


# In[72]:


clean_val_x, clean_val_y = validation_process(test_set, MMscaler)


# In[73]:


clean_val_x.head()


# In[74]:


val_pred = fitted_DT_grid.predict(clean_val_x)
print(classification_report(clean_val_y, val_pred))


# In[75]:


for i, v in enumerate(val_pred):
    if v != clean_val_y[i]:
        if v == 1:
            print('You probably should have won this game')
        elif v == 0:
            print('I am surprised you won!')
    elif v == clean_val_y[i]:
        print('Correct prediction')


# In[77]:


def input_model(data, target, model):
    val_pred = model.predict(data)
    print(classification_report(target, val_pred))
    return(val_pred)
def recommend(val_pred, target):
    for i, v in enumerate(val_pred):
        if v != target[i]:
            if v == 1:
                print('You probably should have won this game')
            elif v == 0:
                print('I am surprised you won!')
        elif v == target[i]:
            print('Correct prediction')
    return None




lux_x, lux_y = full_process('Luxorth', MMscaler)
lux_preds = input_model(lux_x, lux_y, fitted_DT_grid)
recommend(lux_preds, lux_y)






