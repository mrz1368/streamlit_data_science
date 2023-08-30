import streamlit as st
import pandas as pd
import joblib
from keras import models
from spaceship_etl import bool_transformer,col_splitter,path_maker,avg_spent, cleaner
from streamlit_shap import st_shap
import shap


#st.set_page_config(layout="wide")

#load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv')
    return df

# Load the preprocessor from the pickel file
def load_preprocessor():
	filename = 'models/spaceship_ML_model.pkl'
	Ml_Pipeline = joblib.load(filename)
	return Ml_Pipeline['preprocessor']
	


train=load_data()

train['Transported'] = train['Transported'].astype(int)
# feature engineering functions
train = cleaner(train)
train = path_maker(train)
train = avg_spent(train)
#st.write(train.head())

features = ['CryoSleep','Age','VIP','Path','Avg_In_Spent','Avg_Out_Spent','Deck','Cabin_num','Side']
# select the appropriate features  
X_train = train[features]
#st.write(X_train.head())


preprocessor = load_preprocessor()
X_train = preprocessor.transform(X_train)

col_names = []
for i in preprocessor.get_feature_names_out():
    col_names.append(i.split("__")[1])

X_train_df = pd.DataFrame(X_train, columns=col_names)
#st.write(X_df.head())

with st.form(key='my_form'):
	model = st.selectbox("Select a model for prediction:", ['CatBoost','Neural Network Classifier'])

	name = st.text_input(label='Enter the Passenger\'s Name', placeholder="John Doe")
	homeplanet = st.radio("What was the Passenger\'s Home Planet?", ('Earth', 'Europa', 'Mars'),horizontal=True)
	destination = st.radio("What was the Passenger\'s Destination?", ('55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-le'),horizontal=True)
	age = st.slider("Select the Passenger\'s Age:", min_value=1, max_value=100,value=27)
	
	deck = st.selectbox("What was the Passenger\'s Cabin Deck?", ['A','B','C','D','E','G','T'])
	cabin_num = st.slider("Select the Passenger\'s Cabin Number:", min_value=1, max_value=1894,value=422)
	side = st.radio("What was the Passenger\'s Cabin Side?", ('Port', 'Starboard'),horizontal=True)
	side = "P" if side == 'Port' else "S"
	
	cabin = deck+'/'+str(int(cabin_num))+'/'+side
	
	st.write("Select the services the passenger used:")
	col1, col2 = st.columns(2)
	with col1:
		vip = st.checkbox("VIP")
	with col2:
		cryo = st.checkbox("Cryo Sleep")

	RoomService = st.number_input('How much was spent on Room Service?',max_value=15000.00, format='%0.2f', step=1.00)
	FoodCourt = st.number_input('How much was spent on Food Court?',max_value=30000.00, format='%0.2f', step=1.00)
	ShoppingMall = st.number_input('How much was spent on Shopping Mall?', max_value=25000.00, format='%0.2f', step=1.00)
	Spa = st.number_input('How much was spent on Spa?', max_value=25000.00, format='%0.2f', step=1.00)
	VRDeck = st.number_input('How much was spent on VR Deck?', max_value=25000.00, format='%0.2f', step=1.00)

	submit_button = st.form_submit_button(label='Submit')

if submit_button:
	
	#convert our inputs to pandas dataframe
	temp_df = pd.DataFrame(
		data= [['0000_00',homeplanet, cryo, cabin, destination, age, vip,
			RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, name ]],
		columns=['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
    		'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
       		'Name']
	   )

	# feature engineering functions
	temp_df = col_splitter(temp_df)
	temp_df = bool_transformer(temp_df)
	temp_df = path_maker(temp_df)
	temp_df = avg_spent(temp_df)

	# select the appropriate features  
	X_test = temp_df[features]
	
	if model == 'CatBoost':
		# Load the ML model from the pickel file
		filename = 'models/spaceship_ML_model.pkl'
		Ml_Pipeline = joblib.load(filename)

		y_pred = Ml_Pipeline.predict(X_test)
		y_pred = [True if value >= 0.5 else False for value in y_pred]

	
	elif model == 'Neural Network Classifier':
		# Load the model from the h5 file
		modelFileName = 'models/spaceship_DL_classifier.h5'
		DL_model = models.load_model(modelFileName)
		
		# Call the preprocessor from the ML Pipline file
		X_test_transformed = preprocessor.transform(X_test)
		y_pred = DL_model.predict(X_test_transformed)
		y_pred = [True if value >= 0.5 else False for value in y_pred]
	

	result = ':green[Transported]' if y_pred[0] == True else ':red[Not Transported]'
	st.write(f"Passenger {name} is {result}")

	X_test_transformed = preprocessor.transform(X_test)
	X_df = pd.DataFrame(X_test_transformed, columns=col_names)
	
	st.header('SHAP output')
	with st.expander('SHAPLY Force plot'):
		if model == 'CatBoost':	
			# Create object that can calculate shap values
			explainer = shap.TreeExplainer(Ml_Pipeline['model'])
			shap_values = explainer.shap_values(X_df)		
			# visualize the shap plot
			st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_df.iloc[0,:]), height=200, width=650)
		
		elif model == 'Neural Network Classifier':
			explainer = shap.KernelExplainer(DL_model, X_train[:50,:])
			shap_values = explainer.shap_values(X_df, nsamples=500)
			st_shap(shap.force_plot(explainer.expected_value, shap_values[0], X_df.iloc[0,:]), height=200, width=650)		
