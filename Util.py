import pandas as pd
import numpy as np
from riotwatcher import LolWatcher, ApiError
import time
from datetime import datetime
import warnings
import streamlit as st
warnings.filterwarnings('ignore')
random_state = 343

api_key = 'RGAPI-6f0e51af-d969-4e67-978e-1cf2048197ae'
watcher = LolWatcher(api_key)
my_region = 'na1'


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
    player_data = pd.DataFrame()
    start = time.time()
    summonerName_list = []
    participant_list = []
    team_id = []
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
            for participant in req['participantIdentities']:
                ph_part_df = pd.DataFrame.from_dict(participant)
                ph_sum_name_df = pd.DataFrame.from_dict(participant['player'], orient = 'index')
                ph_sum_name_df = ph_sum_name_df.T
                participant_list.append(ph_part_df['participantId'])
                # st.write(participant_list)
                for part in ph_sum_name_df['summonerName']:
                    summonerName_list.append(part)
            for i in participant_list:
                participant_data_df = pd.DataFrame.from_dict(req['participants'])
                team_id.append(participant_data_df['teamId'])
                st.write(participant_data_df['teamId'])
                #     # for player in playerDto['player']['SummonerName']:
                #     ph_player_df = pd.DataFrame.from_dict(ph_part_df['player'])
                #     summonerName_list.append(ph_player_df['player'][7])
                # for teamId in ph_part_df['participants']['teamId']:
    participant_df = pd.DataFrame()
    participant_df['summonerName']= pd.Series(summonerName_list)
    participant_df['participantId']= pd.Series(participant_list)
    participant_df['teamId']= pd.Series(team_id)
    end = time.time()
    print(end-start)
    return team_data, participant_df
# preprocessing functions and single data grab
def call_games(summonerName, games):
    '''
    Calls RIOT API for team data from a single summoner's name
    ARGS:
        summonerName: Name of summoner we need information on
            TYPE: String
    returns DataFrame of last 10 games' team data
    '''
    games = int(games)
    req = watcher.summoner.by_name(my_region, summonerName)
    account_list = []
    account_list.append(req['accountId'])
    matches = get_gameId_from_accId(account_list, 1)
    team_data, summonerName_df = team_data_from_match(matches[:games], 1)
    st.write(summonerName_df)
    team_data['summonerName'] = pd.Series(summonerName_list)
    team_data_summoner = team_data['summonerName'] == summonerName
    return(team_data, team_data_summoner)

def clean_data(team_data):
    # team_data = team_data.iloc[::5, :]
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

def displayable_data(team_data_summoner):
    team_data_summoner['win'] = team_data_summoner['win'].map(lambda x: 'Win' if x == 1 else 'Lost')
    # team_data = call_games(summoner, games)[1]
    for col in team_data_summoner.columns:
        if col == 'index':
            team_data_summoner.drop(colums = col, inplace = True)
    team_data_summoner.reset_index()
    return(team_data_summoner)

def scale_data(team_data, scaler):
    for col in team_data.columns:
        if col == 'index':
            team_data.drop(columns = col, inplace = True)
        if col == 'summonerName':
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

def full_process(summonerName, scaler, games):
    team_data, team_data_summoner = call_games(summonerName, games)
    team_data_clean = clean_data(team_data)
    team_data_x, team_data_y = process_data(team_data_clean)
    team_data_raw= displayable_data(team_data_summoner)
    team_data_x = scale_data(team_data_x, scaler)
    return(team_data_x, team_data_y, team_data_raw)

def validation_process(team_data, scaler):
    team_data_clean = clean_data(team_data)
    team_data_x, team_data_y = process_data(team_data_clean)
    team_data_x = scale_data(team_data_x, scaler)
#     for col in team_data_x.columns:
#         if col == 'index':
#             team_data_x.drop(columns = col, inplace = True)
    return(team_data_x, team_data_y)

def input_model(data, target, model):
    val_pred = model.predict(data)
    classification_report(target, val_pred)
    return(val_pred)

def recommend(val_pred, target):
    for i, v in enumerate(val_pred):
        if v != target[i]:
            if v == 1:
                'You probably should have won this game'
            elif v == 0:
                'I am surprised you won!'
        elif v == target[i]:
            'Correct prediction'
    return None