from Util import *
import streamlit as st
import joblib
from sklearn.metrics import classification_report

st.title('League of Legends: Should I have won that game?')

user_input = st.text_input('Please input summoner name')

user_input

# Add a selectbox to the sidebar:
games = st.sidebar.text_input(
    'How many games would you like to analyze? (Up to 5 games)',
    )

scaler = joblib.load('Min_Max_scaler.gz')
model = joblib.load('Decision_Tree.gz')
if st.button('Search'):
	if int(games) > 20:
		st.sidebar.write('Please only analyze up to 5 games at a time.', font='red')
	else:
		summoner_data_x, summoner_data_y, summoner_data_raw= full_process(user_input, scaler, games)
		game_predictions = input_model(summoner_data_x, summoner_data_y, model)
		for i, pred in enumerate(game_predictions):
			st.write(summoner_data_raw.iloc[i])
			recommend(game_predictions[i], summoner_data_y[i], summoner_data_raw, i)
			