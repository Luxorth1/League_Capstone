from Util import *
import streamlit as st
import joblib
from sklearn.metrics import classification_report

st.title('League of Legends: Should I have won that game?')

st.write('This app will allow you to input your League of Legends Summoner name and the number of \
	desired games to analyze. Once this input is submitted it will grab your data from the \
	Riot servers for your previous games based on the number you entered. From here it will \
	pump your data into a fancy machine learning algorithm and determine if you should have won the game.\
	')

st.write('If the model determines you should have won it will provide recommendations based on \
	objectives that the algorithm deems more important. This model has been trained with over\
	30,000 match outcomes from top ranked players.')


user_input = st.text_input('Please input summoner name')

user_input

# Add a selectbox to the sidebar:
games = st.sidebar.text_input(
    'How many games would you like to analyze? (Please keep this number below 10)',
    )
if games > 10:
	st.write('PLEASE KEEP THIS NUMBER BELOW 10')
else:
	scaler = joblib.load('Min_Max_scaler.gz')
	model = joblib.load('Decision_Tree.gz')
	if st.button('Search'):
		try:
			if int(games) > 20:
				st.sidebar.write('Please only analyze up to 5 games at a time.', font='red')
			else:
				summoner_data_x, summoner_data_y, summoner_data_raw= full_process(user_input, scaler, games)
				game_predictions = input_model(summoner_data_x, summoner_data_y, model)
				for i, pred in enumerate(game_predictions):
					st.write(summoner_data_raw.iloc[i])
					recommend(game_predictions[i], summoner_data_y[i], summoner_data_raw, i)
		except:
			st.write('Please validate that you have spelt your Summoner name correctly.')
			