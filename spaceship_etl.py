from sklearn.impute import SimpleImputer, KNNImputer

def col_splitter(df):
    #PassengerId A is unique Id for each passenger. 
    #Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group.
    df['GroupId'] = [int(i.split('_')[0]) for i in df['PassengerId']]
    
    #The cabin number takes the form deck/num/side, where side can be either P for Port or S for Starboard
    df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    
    return df

def bool_transformer(df):
    df['VIP'] = df['VIP'].astype(int)
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    return df

def custom_imputer(df):

    # create an instance of the SimpleImputer class
    imputer = SimpleImputer(strategy='most_frequent')

    # find the group ID of passengers whose 'HomePlanet' or 'Destination' value is missing
    group_list = df.loc[(df['Destination'].isna()) | (df['HomePlanet'].isna())]['GroupId'].to_list()
    
    # we know for sure passengers within a group have the same HomePlanet and Destination, so we will apply imputer within groups.
    for i in group_list:    
        temp_df = df.loc[df.GroupId == i].copy()
        idx = temp_df.index
        group_size = temp_df.shape[0]
        # check whether whole data within the group is missing or not
        if group_size > 1 and temp_df['HomePlanet'].isna().sum() < group_size and temp_df['Destination'].isna().sum() < group_size:
            df.loc[idx,['HomePlanet', 'Destination']] = imputer.fit_transform(temp_df[['HomePlanet', 'Destination']])
            
    # as we saw the Deck passengers use to get on spaceship depend on the starting point of their travel,
    # for example there is no Deck 'E', 'F', 'G' in Europa. 
    # we will impute the missing data of Deck column within each HomePlanet seperately.
    HomePlanets = ['Europa', 'Earth', 'Mars']
    for planet in HomePlanets:    

        temp_df = df.loc[df.HomePlanet == planet].copy()
        idx = temp_df.index

        df.loc[idx,'Deck'] = imputer.fit_transform(temp_df[['Deck']])


    # take the same aproach for imputing Deck columns based on Destinations.
    Destinations = ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e']    
    for Destination in Destinations:    

        temp_df = df.loc[df.Destination == Destination].copy()
        idx = temp_df.index

        df.loc[idx,'Deck'] = imputer.fit_transform(temp_df[['Deck']])


    # Now we don't have any missing Deck Value, So we can impute impute all the remaining HomePlanet and Destination of
    # the passengers who travel alone or whole data of the group was missing. 
    Decks = ['B', 'F', 'A', 'G', 'E', 'D', 'C', 'T']
    for Deck in Decks:    

        temp_df = df.loc[df.Deck == Deck].copy()
        idx = temp_df.index

        df.loc[idx,['HomePlanet', 'Destination']] = imputer.fit_transform(temp_df[['HomePlanet', 'Destination']])
        
    
    # We will impute all numerical and boolean values with KNN Imputer because it is likely the passengers who are neighbours 
    # have the same behaviour spending or age
    KNNimputer = KNNImputer(n_neighbors=2, weights="uniform")

    features_num = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 'CryoSleep', 'VIP']

    df[features_num] = KNNimputer.fit_transform(df[features_num])
    
    return df

def cleaner(df):
    df = col_splitter(df)
    df = custom_imputer(df)
    df.Side = df.Side.fillna('P')
    df = bool_transformer(df)
    return df

def path_maker(df):
    # Define a lambda function to show the travel path
    path_maker = lambda x, y: x + "-" + y

    # Use the map function to apply the add function to the 'Value1' and 'Value2' columns and create a new column 'Path'
    df['Path'] = df[['HomePlanet','Destination']].apply(lambda x: path_maker(*x), axis=1)
    
    return df

def avg_spent(df):
    #df['Avg_Spent'] = df[['RoomService', 'Spa', 'VRDeck', 'FoodCourt', 'ShoppingMall']].sum(axis=1)/5
    df['Avg_In_Spent'] = df[['RoomService', 'Spa', 'VRDeck']].sum(axis=1)/3
    df['Avg_Out_Spent'] = df[['FoodCourt', 'ShoppingMall']].sum(axis=1)/2
    return df