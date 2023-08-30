import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv')
    return df


train=load_data()
    
st.header("Preveiw Data")
st.subheader("Data Sample")
st.write(train.head(5))
st.write(f"Number of rows in **training** dataset: :green[{train.shape[0]}]")

chekbox = st.checkbox('Columns Info')
if chekbox:
    st.markdown("""
- **PassengerId**: A unique Id for each passenger. Each Id takes the form _gggg_pp_ where _gggg_ indicates a group the passenger is travelling with and _pp_ is their number within the group. People in a group are often family members, but not always.
- **HomePlanet**: The planet the passenger departed from, typically their planet of permanent residence.
- **CryoSleep**: Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
- **Cabin**: The cabin number where the passenger is staying. Takes the form _deck/num/side_, where _side_ can be either _P_ for _Port_ or _S_ for _Starboard_.
- **Destination**: The planet the passenger will be debarking to.
- **Age**: The age of the passenger.
- **VIP**: Whether the passenger has paid for special VIP service during the voyage.
- **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**: Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
- **Name**: The first and last names of the passenger.
- **Transported**: Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
""")

st.subheader("Statistics")    

options = st.multiselect(
    'Select Columns:',
    train.columns
)
for option in options:
    if option in ['RoomService','VRDeck','Spa','ShoppingMall','FoodCourt']:
        fig, ax = plt.subplots()
        sns.histplot(data=train, x=option, bins=50)
        st.write(f"Distribution of  values in the **{option}** column:")
        st.pyplot(fig)
        st.write(f'Number of missing values in the **{option}** column: **:red[{train[option].isnull().sum()}]**')

    elif option == 'Age':
        fig, ax = plt.subplots()
        sns.histplot(data=train, x= 'Age', binwidth=5)
        st.write(f"Distribution of  values in the **{option}** column:")
        st.pyplot(fig)
        st.write(f'Number of missing values in the **{option}** column: **:red[{train[option].isnull().sum()}]**')

    elif option in ['CryoSleep','Transported', 'HomePlanet', 'Destination', 'VIP']:
        fig, ax = plt.subplots()
        train[option].value_counts(dropna=False).plot.pie( startangle=180, autopct='%1.1f%%', pctdistance=0.6, labeldistance=1.1)
        st.write(f"Distribution of  values in the **{option}** column:")
        st.pyplot(fig)
    else:
        st.metric(f'Number of missing values in the **{option}** column:',train[option].isnull().sum())

