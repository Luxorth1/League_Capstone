import pandas as pd
import numpy as np
from riotwatcher import LolWatcher, ApiError
import time
from datetime import datetime
import warnings
import streamlit as st
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')
random_state = 343

api_key = 'RGAPI-82531dcb-3e63-4ecd-abe3-c3607b363e31'
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
                participant_data_df.T
                team_id.append(participant_data_df['teamId'])
    participant_df = pd.DataFrame()
    participant_df['summonerName']= pd.Series(summonerName_list)
    participant_df['participantId']= pd.Series(participant_list)
    participant_df['teamId']= pd.Series(team_id)
    # st.write(participant_df['participantId'][1]['summonerId'])
    # st.write(participant_df['teamId'][0])
    end = time.time()
    # grab_summoner_game_data(participant_df, 'Luxorth')
    print(end-start)
    return team_data, participant_df

# preprocessing functions and single data grab
def grab_summoner_game_data(Name_df, summonerName):
    team_df = pd.DataFrame()
    participantId_list = []
    teamId_list = []
    summoner_rows = Name_df
    summoner_rows.reset_index(inplace = True)
    for row in summoner_rows['participantId']:
        participantId_list.append(row['accountId'])
    for part in participantId_list:
        if part <= 5 :
            teamId_list.append(100)
        else:
            teamId_list.append(200)
    summoner_rows['participantId'] = pd.Series(participantId_list)
    summoner_rows['teamId'] = pd.Series(teamId_list)
    summoner_rows.drop(columns = 'index', inplace = True)
    return summoner_rows

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
    team_data.reset_index(inplace= True)
    summonerName_df = team_data.join(summonerName_df, rsuffix = 'drop_')
    summonerName_df.drop(columns = ['bans', 'dominionVictoryScore', 'vilemawKills', 
        'index', 'teamIddrop_'], inplace = True)
    return(team_data, summonerName_df) 

def displayable_data(team_data_summoner, games, summonerName):
    team_data_summoner  = grab_summoner_game_data(team_data_summoner, summonerName)
    # team_data = call_games(summoner, games)[1]
    team_data_summoner.reset_index()
    team_data_summoner['win'] = team_data_summoner['win'].map(lambda x: 'Win' if x == 'Win' else 'Lost')
    team_data_summoner = team_data_summoner[team_data_summoner['summonerName'] == summonerName]
    team_data_summoner['teamId'] = team_data_summoner['teamId'].apply(lambda x: 'Blue' if x == 100 else 'Red')
    team_data_summoner.reset_index(inplace = True)
    team_data_summoner.drop(columns = 'index', inplace = True)
    column_order = ['summonerName', 'teamId', 'win', 'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon',
        'firstRiftHerald', 'towerKills', 'inhibitorKills', 'dragonKills', 'riftHeraldKills', 'baronKills']
    team_data_summoner = team_data_summoner[column_order]
    return(team_data_summoner)

def scale_data(team_data, scaler, summonerName):
    for col in team_data.columns:
        if col == 'index':
            team_data.drop(columns = col, inplace = True)
        if col == 'level_0':
            team_data.drop(columns = col, inplace = True)
        if col == 'participantId':
            team_data.drop(columns = col, inplace = True)
    team_data = team_data[team_data['summonerName'] == summonerName]
    team_data.reset_index(inplace = True)
    x_num_columns = ['towerKills', 'inhibitorKills', 'dragonKills', 'riftHeraldKills', 'baronKills']
    x_cat_columns = ['teamId', 'firstBlood','firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon',
        'firstRiftHerald', 'summonerName', 'win']
    X_num = team_data[x_num_columns].copy(deep=True)
    X_cat = team_data[x_cat_columns].copy(deep=True)
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
    team_data, team_data_summoners = call_games(summonerName, games)
    team_data_raw = displayable_data(team_data_summoners, games, summonerName)
    team_data_x = scale_data(team_data_summoners, scaler, summonerName)
    team_data_y = team_data_x['win']
    team_data_y = team_data_y.map(lambda x: 1 if x =='Win' else 0)
    team_data_x.drop(columns = ['win', 'summonerName'], inplace = True)
    return(team_data_x, team_data_y, team_data_raw)

def input_model(data, target, model):
    val_pred = model.predict(data)
    return(val_pred)

def recommend(val_pred, target, team_data_raw, iteration):
    if val_pred != target:
        if val_pred == 1:
            st.write('You probably should have won this game')
            inhibitorKills = team_data_raw.iloc[iteration, 10]
            firstTower = team_data_raw.iloc[iteration, 4]
            dragonKills = team_data_raw.iloc[iteration, 11]
            if inhibitorKills < 1:
                st.write('Recommendation: You should focus on lane objectives more. Downing Towers \
                    leads to more opportunity to get Inhbitors which allows you more lane pressure \
                    from Super Minions.')
            elif firstTower is False:
                st.write('Recommendation: You should focus on making sure you have good lane pressure,\
                    this allows you to secure an early tower. Towers provide gold for the whole team, \
                    causing your team to pull ahead as far as item builds go. This leads to an overall\
                    better advantage over the opponent.')
            elif dragonKills < 1:
                st.write('Recommendation: You should focus on trying to fight for more Dragon kills, \
                    this provides gold and buff to your team, extending your advantage from other objectives.')
        elif val_pred == 0:
            st.write('I am surprised you won!')
    elif val_pred == target:
        st.write('Correct prediction')
    # else:
    #     for i, v in enumerate(val_pred):
    #         if v != target[i]:
    #             if v == 1:
    #                 st.write('You probably should have won this game')
    #             elif v == 0:
    #                 st.write('I am surprised you won!')
    #         elif v == target[i]:
    #             st.write('Correct prediction')
    return None