import streamlit as st
import pandas as pd
from spaceship_etl import cleaner, path_maker, avg_spent
import matplotlib.pyplot as plt
import seaborn as sns

#load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv')
    return df

train=load_data()
# st.write(train.head())
st.header("Compound Features")

st.subheader("1. Grouping amenity expenditures")
st.text('We separated the amenity expenditures into two groups:\n \
        Avg_In_Spent: Average of expenditures of “RoomService”, “VRDeck”, “Spa”.\n \
        Avg_Out_Spent: Average of expenditures of “ShoppingMall”, “FoodCourt”.')
st.text("It has a significant impact on Accuracy of our model around 0.05")

fig, ax = plt.subplots(2,3,  figsize=(14, 8))
sns.histplot(data=train, x= 'RoomService', hue='Transported', bins=50, ax=ax[0,0]);
sns.histplot(data=train, x= 'VRDeck', hue='Transported', bins=50, ax=ax[0,1]);
sns.histplot(data=train, x= 'Spa', hue='Transported', bins=50, ax=ax[0,2]);
sns.histplot(data=train, x='ShoppingMall', hue='Transported', bins=50, ax=ax[1,0]);
sns.histplot(data=train, x='FoodCourt', hue='Transported', bins=50, ax=ax[1,1])
# remove the last subplot in the second row
fig.delaxes(ax[1, 2])
st.pyplot(fig)
st.caption('There is a disparity in the spending category and their impact on being transported.')


train = avg_spent(train)
fig1, ax1 = plt.subplots(1,2, figsize=(14, 7))
sns.histplot(data=train, x='Avg_In_Spent', hue='Transported', ax=ax1[0], bins=50)
sns.histplot(data=train, x='Avg_Out_Spent', hue='Transported', ax=ax1[1], bins=50)
st.pyplot(fig1)
st.caption('It seems \"Avg_In_Spent\" has significant impact on being transported')

st.subheader("2. Path")
st.text("We will use this compound feature of \"Path\" instead of ”Home” and “Destination”\nfor our modeling, because it gives of a better outcome in terms of Accuracy Score.")

fig2, ax2 = plt.subplots(1,2, figsize=(14, 7))
sns.countplot(data=train, x='HomePlanet', hue='Transported', palette='tab10', ax=ax2[0])
ax2[0].set_xticklabels(ax2[0].get_xticklabels(), rotation=90)
sns.countplot(data=train, x='Destination', hue='Transported', palette='tab10', ax=ax2[1])
ax2[1].set_xticklabels(ax2[1].get_xticklabels(), rotation=90)
st.pyplot(fig2)
st.caption('There is no significant relationship between Home Planet or Destination with being transported.')

train['Transported'] = train['Transported'].astype(int)
train = cleaner(train)
train = path_maker(train)


#Create count plot of path with Transported as hue
fig3, ax3 = plt.subplots(figsize=(14, 7))
sns.countplot(data=train, x='Path', hue='Transported', palette='tab10', ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
st.pyplot(fig3)
st.caption('It seems being transported had a more meaningful relationship to the whole path passengers \
           took in comparison with \'home\' and \'destination\' separately.')


# import joblib
# features = ['CryoSleep','Age','VIP','Path','Avg_In_Spent','Avg_Out_Spent','Deck','Cabin_num','Side']
# # select the appropriate features  
# X_train = train[features]
# st.write(X_train.head())

# # Load the ML model from the pickel file
# filename = 'models/spaceship_ML_model.pkl'
# Ml_Pipline = joblib.load(filename)

# X_transformed = Ml_Pipline['preprocessor'].transform(X_train)
# names = []
# for i in Ml_Pipline['preprocessor'].get_feature_names_out():
#     names.append(i.split("__")[1])
# X_df = pd.DataFrame(X_transformed, columns=names)
# st.write(X_df.head())
