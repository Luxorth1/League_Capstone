from Util import *
import streamlit as st
import joblib

user_input = st.text_input('Please input summoner name')

user_input

# Add a selectbox to the sidebar:
games = st.sidebar.text_input(
    'How many games would you like to analyze?',
    )

scaler = joblib.load('Min_Max_scaler.gz')
model = joblib.load('Decision_Tree.gz')
if st.button('Search'):
	# try:
		summoner_data_x, summoner_data_y, summoner_data_raw= full_process(user_input, scaler, games)
		# st.write(summoner_data_raw)
	# except:
	# 	'Please make sure you have entered a Summoner Name.'


