import pandas as pd
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing



def User_item_matrix(df4):
    df4.drop(df4.columns[[1,2,3,4]], 1, inplace=True)
    return df4



def PredictionsCB(UserID, df):
    UserID=UserID-1
    df1=df[['Gender', 'Age', 'Personality']]
    model = NearestNeighbors(n_neighbors = 11).fit(df1)
    distances, indices = model.kneighbors(df1)
    
    indices = pd.DataFrame(indices)
    df2=User_item_matrix(df.copy())
    l=[]
    for i in df2.columns[1:]:
        summ = 0
        avg=0
        k=0
        for j in indices.iloc[UserID]:
            if j == UserID:
                k=k
                continue 
            else: 
                summ =summ + df2[i][j]
                k+=1
        avg = summ/k
        l.append(avg)
    return l



def PredictionsCF(UserID,df):
    
    df3=User_item_matrix(df.copy())
    
    df3 = pd.melt(df3, id_vars='ID', var_name='Activity', value_name='Rating')
    df3= df3.fillna(df3.Rating.mean())
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df3[['ID','Activity', 'Rating']], reader)

    df3= df3.fillna(df3.Rating.mean())
    
    trainset, testset = train_test_split(data, test_size = 0.2)

    model = SVD()

    # Run 5-fold cross-validation and print results.
    cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True);
    
    model.fit(trainset)
    
    predictions = model.test(testset)

    Recommendations = []
    for i in df3.Activity.unique():
        pred = model.predict(UserID, str(i))
        Recommendations.append(pred.est)
            
    return Recommendations, accuracy.rmse(predictions)


def createNewUser(Age, Gender, Personality, df):
    UserID=len(df)+1
    UserInput = {'ID': [UserID], "Age":[Age], "Gender":[Gender],'Personality': [Personality], 'NoR': [0]}
    newuser = pd.DataFrame(UserInput)
    df=df.append(newuser)
    df5=df
    CB=PredictionsCB(UserID, df5)
    #Replace NaNs by values in CB
    for i in range(5, len(df.columns)):
        df.iat[UserID-1, i] = CB[i-5]
    return df

def FinalPredictions(listCB, listCF,df10, threshold, UserID):
    listfinal= []
    ratings = df10.iloc[UserID-1]['NoR']
    if ratings <=5:
        for i in range(len(listCB)):
            if listCB[i] >=threshold:
                listfinal.append(df10.columns[i+5])
    elif ratings <= 15 and ratings> 5:
        for i in range(len(listCB)):
            if (listCB[i] + listCF[i])/2 >=threshold:
                listfinal.append(df10.columns[i+5])
    else:
        for i in range(len(listCF)):
            if listCF[i] >=threshold:
                listfinal.append(df10.columns[i+5])   
    return listfinal
    
def ratingInput(UserID, Activity, Rating, df):
    i=df.columns.get_loc(str(Activity))
    df.iat[UserID-1, i] =Rating
    #update Number of Ratings
    df.iat[UserID-1,4]=df.iat[UserID-1,4]+1
    return df
         
            
       

# df_sad = pd.read_csv(r'C:/Users/User/Desktop/490/HAPPY.csv', encoding= 'unicode_escape')          
# df_sad=createNewUser(60, 0, 2, df_sad)  
# df_sad = ratingInput(301, 'Gym/Workout Session (10-60min)',0, df_sad)     


# l2=PredictionsCB(301,df_sad)
# l1,RMSE=PredictionsCF(301, df_sad)
# print(FinalPredictions(l2,l1,df_sad,301))


"""PLEASE KEEP IN MY MIND THAT THE BELOW IS TESTING THAT SIMULATES REAL USE, 
PLEASEV CONSIDERING LIMITATIONS DUE TO EXTENSIVE NEED OF UI WORK/ONLY FOCUS ON ML"""
#We will be testing with the broadest user sample possible

################################################################ 

#In the case the user feeling is ANXIOUS:
# New User is : 20 year old male, who is a thinker.
df_anxious = pd.read_csv(r'C:/Users/User/Desktop/MOODS/ANXIOUS.csv', encoding= 'unicode_escape')  
df_anxious=createNewUser(20, 1, 2, df_anxious) 
#Cold start
l2=PredictionsCB(301,df_anxious)
l1,RMSE=PredictionsCF(301, df_anxious)
print("ANXIOUS:\n","Cold start:\n",FinalPredictions(l2,l1,df_anxious,0.3, 301))
#After Decent use and rating:
df_anxious = ratingInput(301, 'Gym/Workout Session (10-60min)',1, df_anxious) 
df_anxious = ratingInput(301, 'Listening to Music',1, df_anxious) 
df_anxious = ratingInput(301, 'Call/See/Hug friends',1, df_anxious) 
df_anxious = ratingInput(301, 'Call/See/Hug family members',1, df_anxious) 
df_anxious = ratingInput(301, 'Go out for dinner/club/bar',0, df_anxious) 
df_anxious = ratingInput(301, 'Go out for a Walk/Run/Drive',1, df_anxious) 
df_anxious = ratingInput(301, 'Study/Learn something new',1, df_anxious) 
df_anxious = ratingInput(301, 'Eat/Cook/Prepare a Meal',1, df_anxious) 

l2=PredictionsCB(301,df_anxious)
l1,RMSE=PredictionsCF(301, df_anxious)
print("After Some Use:\n",FinalPredictions(l2,l1,df_anxious,0.4, 301))
#After Extensive Use
df_anxious = ratingInput(301, 'Watch Action Movie/Series',0, df_anxious) 
df_anxious = ratingInput(301, 'Breathing Exercises',1, df_anxious) 
df_anxious = ratingInput(301, 'Yoga/Meditation',1, df_anxious) 
df_anxious = ratingInput(301, 'Surf Social Media/YouTube (staying time conscious!)',1, df_anxious) 
df_anxious = ratingInput(301, 'Watch Romantic Movie/Series',0, df_anxious) 
df_anxious = ratingInput(301, 'Take/Plan a vacation or travel',1, df_anxious) 
df_anxious = ratingInput(301, 'Get more sleep/Take a nap',0, df_anxious) 
df_anxious = ratingInput(301, 'Drink some coffee (moderately)',1, df_anxious)
df_anxious = ratingInput(301, 'Get Creative: Draw, Paint, Play an instrument',1, df_anxious) 
df_anxious = ratingInput(301, 'Go shopping',1, df_anxious) 
df_anxious = ratingInput(301, 'Watch Comedy Movie/Series',0, df_anxious) 
df_anxious = ratingInput(301, 'Create a schedule/Organize your time/Declutter',1, df_anxious) 
df_anxious = ratingInput(301, 'Gaming (staying time conscious!)',0, df_anxious) 
df_anxious = ratingInput(301, 'Watch Action Movie/Series',0, df_anxious) 
df_anxious = ratingInput(301, 'Reading',1, df_anxious) 
df_anxious = ratingInput(301, "Write down how you feel/what you're grateful for",1, df_anxious)
l2=PredictionsCB(301,df_anxious)
l1,RMSE=PredictionsCF(301, df_anxious)
print("After Extensive Use:\n",FinalPredictions(l2,l1,df_anxious,0.6, 301))


########################################################3
df_sad = pd.read_csv(r'C:/Users/User/Desktop/MOODS/SAD.csv', encoding= 'unicode_escape')
df_sad=createNewUser(20, 1, 2, df_sad) 
#Cold start
l2=PredictionsCB(301,df_sad)
l1,RMSE=PredictionsCF(301, df_sad)
print("\n\n\nSAD:\n","Cold start:\n",FinalPredictions(l2,l1,df_sad, 0.45, 301))
#After Decent use and rating:
df_sad = ratingInput(301, 'Gym/Workout Session (10-60min)',0, df_sad) 
df_sad = ratingInput(301, 'Listening to Music',1, df_sad) 
df_sad = ratingInput(301, 'Call/See/Hug friends',0, df_sad) 
df_sad = ratingInput(301, 'Call/See/Hug family members',0, df_sad) 
df_sad = ratingInput(301, 'Go out for dinner/club/bar',0, df_sad) 
df_sad = ratingInput(301, 'Go out for a Walk/Run/Drive',1, df_sad) 
df_sad = ratingInput(301, 'Study/Learn something new',1, df_sad) 
df_sad = ratingInput(301, 'Eat/Cook/Prepare a Meal',1, df_sad) 

l2=PredictionsCB(301,df_sad)
l1,RMSE=PredictionsCF(301, df_sad)
print("After Some Use:\n",FinalPredictions(l2,l1,df_sad,0.45, 301))
#After Extensive Use
df_sad = ratingInput(301, 'Watch Action Movie/Series',0, df_sad) 
df_sad = ratingInput(301, 'Breathing Exercises',1, df_sad) 
df_sad = ratingInput(301, 'Yoga/Meditation',1, df_sad) 
df_sad = ratingInput(301, 'Surf Social Media/YouTube (staying time conscious!)',1, df_sad) 
df_sad = ratingInput(301, 'Watch Romantic Movie/Series',0, df_sad) 
df_sad = ratingInput(301, 'Take/Plan a vacation or travel',1, df_sad) 
df_sad = ratingInput(301, 'Get more sleep/Take a nap',0, df_sad) 
df_sad = ratingInput(301, 'Drink some coffee (moderately)',0, df_sad)
df_sad = ratingInput(301, 'Get Creative: Draw, Paint, Play an instrument',0, df_sad) 
df_sad = ratingInput(301, 'Go shopping',0, df_sad) 
df_sad = ratingInput(301, 'Watch Comedy Movie/Series',0, df_sad) 
df_sad = ratingInput(301, 'Create a schedule/Organize your time/Declutter',0, df_sad) 
df_sad = ratingInput(301, 'Gaming (staying time conscious!)',0, df_sad) 
df_sad = ratingInput(301, 'Watch Action Movie/Series',0, df_sad) 
df_sad = ratingInput(301, 'Reading',0, df_sad) 
df_sad = ratingInput(301, "Write down how you feel/what you're grateful for",0, df_sad)
l2=PredictionsCB(301,df_sad)
l1,RMSE=PredictionsCF(301, df_sad)
print("After Extensive Use:\n",FinalPredictions(l2,l1,df_sad,0.4, 301))

     
#################################################################
df_happy = pd.read_csv(r'C:/Users/User/Desktop/MOODS/HAPPY.csv', encoding= 'unicode_escape')  

df_happy=createNewUser(20, 1, 2, df_happy) 
#Cold start
l2=PredictionsCB(301,df_happy)
l1,RMSE=PredictionsCF(301, df_happy)
print("\n\n\nHAPPY:\n","Cold start:\n",FinalPredictions(l2,l1,df_happy,0.5, 301))
#After Decent use and rating:
df_happy = ratingInput(301, 'Gym/Workout Session (10-60min)',1, df_happy) 
df_happy = ratingInput(301, 'Listening to Music',1, df_happy) 
df_happy = ratingInput(301, 'Call/See/Hug friends',1, df_happy) 
df_happy = ratingInput(301, 'Call/See/Hug family members',1, df_happy) 
df_happy = ratingInput(301, 'Go out for dinner/club/bar',1, df_happy) 
df_happy = ratingInput(301, 'Go out for a Walk/Run/Drive',0, df_happy) 
df_happy = ratingInput(301, 'Study/Learn something new',1, df_happy) 
df_happy = ratingInput(301, 'Eat/Cook/Prepare a Meal',1, df_happy) 

l2=PredictionsCB(301,df_happy)
l1,RMSE=PredictionsCF(301, df_happy)
print("After Some Use:\n",FinalPredictions(l2,l1,df_happy,0.5, 301))
#After Extensive Use
df_happy = ratingInput(301, 'Watch Action Movie/Series',0, df_happy) 
df_happy = ratingInput(301, 'Breathing Exercises',0, df_happy) 
df_happy = ratingInput(301, 'Yoga/Meditation',0, df_happy) 
df_happy = ratingInput(301, 'Surf Social Media/YouTube (staying time conscious!)',1, df_happy) 
df_happy = ratingInput(301, 'Watch Romantic Movie/Series',0, df_happy) 
df_happy = ratingInput(301, 'Take/Plan a vacation or travel',1, df_happy) 
df_happy = ratingInput(301, 'Get more sleep/Take a nap',0, df_happy) 
df_happy = ratingInput(301, 'Drink some coffee (moderately)',0, df_happy)
df_happy = ratingInput(301, 'Get Creative: Draw, Paint, Play an instrument',1, df_happy) 
df_happy = ratingInput(301, 'Go shopping',1, df_happy) 
df_happy = ratingInput(301, 'Watch Comedy Movie/Series',1, df_happy) 
df_happy = ratingInput(301, 'Create a schedule/Organize your time/Declutter',0, df_happy) 
df_happy = ratingInput(301, 'Gaming (staying time conscious!)',1, df_happy) 
df_happy = ratingInput(301, 'Watch Action Movie/Series',1, df_happy) 
df_happy = ratingInput(301, 'Reading',0, df_happy) 
df_happy = ratingInput(301, "Write down how you feel/what you're grateful for",0, df_happy)
l2=PredictionsCB(301,df_happy)
l1,RMSE=PredictionsCF(301, df_happy)
print("After Extensive Use:\n",FinalPredictions(l2,l1,df_happy,0.6, 301))




#################################################################
df_angry = pd.read_csv(r'C:/Users/User/Desktop/MOODS/ANGRY.csv', encoding= 'unicode_escape')

df_angry=createNewUser(20, 1, 2, df_angry) 
#Cold start
l2=PredictionsCB(301,df_angry)
l1,RMSE=PredictionsCF(301, df_angry)
print("\n\n\nANGRY:\n","Cold start:\n",FinalPredictions(l2,l1,df_angry,0.3, 301))
#After Decent use and rating:
df_angry = ratingInput(301, 'Gym/Workout Session (10-60min)',1, df_angry) 
df_angry = ratingInput(301, 'Listening to Music',1, df_angry) 
df_angry = ratingInput(301, 'Call/See/Hug friends',0, df_angry) 
df_angry = ratingInput(301, 'Call/See/Hug family members',0, df_angry) 
df_angry = ratingInput(301, 'Go out for dinner/club/bar',0, df_angry) 
df_angry = ratingInput(301, 'Go out for a Walk/Run/Drive',1, df_angry) 
df_angry = ratingInput(301, 'Study/Learn something new',0, df_angry) 
df_angry = ratingInput(301, 'Eat/Cook/Prepare a Meal',0, df_angry) 

l2=PredictionsCB(301,df_angry)
l1,RMSE=PredictionsCF(301, df_angry)
print("After Some Use:\n",FinalPredictions(l2,l1,df_angry,0.3, 301))
#After Extensive Use
df_angry = ratingInput(301, 'Watch Action Movie/Series',0, df_angry) 
df_angry = ratingInput(301, 'Breathing Exercises',1, df_angry) 
df_angry = ratingInput(301, 'Yoga/Meditation',1, df_angry) 
df_angry = ratingInput(301, 'Surf Social Media/YouTube (staying time conscious!)',1, df_angry) 
df_angry = ratingInput(301, 'Watch Romantic Movie/Series',0, df_angry) 
df_angry = ratingInput(301, 'Take/Plan a vacation or travel',0, df_angry) 
df_angry = ratingInput(301, 'Get more sleep/Take a nap',1, df_angry) 
df_angry = ratingInput(301, 'Drink some coffee (moderately)',0, df_angry)
df_angry = ratingInput(301, 'Get Creative: Draw, Paint, Play an instrument',1, df_angry) 
df_angry = ratingInput(301, 'Go shopping',0, df_angry) 
df_angry = ratingInput(301, 'Watch Comedy Movie/Series',1, df_angry) 
df_angry = ratingInput(301, 'Create a schedule/Organize your time/Declutter',0, df_angry) 
df_angry = ratingInput(301, 'Gaming (staying time conscious!)',1, df_angry) 
df_angry = ratingInput(301, 'Watch Action Movie/Series',1, df_angry) 
df_angry = ratingInput(301, 'Reading',0, df_angry) 
df_angry = ratingInput(301, "Write down how you feel/what you're grateful for",0, df_angry)
l2=PredictionsCB(301,df_angry)
l1,RMSE=PredictionsCF(301, df_angry)
print("After Extensive Use:\n",FinalPredictions(l2,l1,df_angry,0.5, 301))




############################################################
df_bored = pd.read_csv(r'C:/Users/User/Desktop/MOODS/BORED.csv', encoding= 'unicode_escape') 
df_bored=createNewUser(20, 1, 2, df_bored) 
#Cold start
l2=PredictionsCB(301,df_bored)
l1,RMSE=PredictionsCF(301, df_bored)
print("\n\n\nBORED:\n","Cold start:\n",FinalPredictions(l2,l1,df_bored,0.5, 301))
#After Decent use and rating:
df_bored = ratingInput(301, 'Gym/Workout Session (10-60min)',1, df_bored) 
df_bored = ratingInput(301, 'Listening to Music',0, df_bored) 
df_bored = ratingInput(301, 'Call/See/Hug friends',1, df_bored) 
df_bored = ratingInput(301, 'Call/See/Hug family members',1, df_bored) 
df_bored = ratingInput(301, 'Go out for dinner/club/bar',1, df_bored) 
df_bored = ratingInput(301, 'Go out for a Walk/Run/Drive',1, df_bored) 
df_bored = ratingInput(301, 'Study/Learn something new',1, df_bored) 
df_bored = ratingInput(301, 'Eat/Cook/Prepare a Meal',1, df_bored) 

l2=PredictionsCB(301,df_bored)
l1,RMSE=PredictionsCF(301, df_bored)
print("After Some Use:\n",FinalPredictions(l2,l1,df_bored,0.5, 301))
#After Extensive Use
df_bored = ratingInput(301, 'Watch Action Movie/Series',1, df_bored) 
df_bored = ratingInput(301, 'Breathing Exercises',0, df_bored) 
df_bored = ratingInput(301, 'Yoga/Meditation',0, df_bored) 
df_bored = ratingInput(301, 'Surf Social Media/YouTube (staying time conscious!)',1, df_bored) 
df_bored = ratingInput(301, 'Watch Romantic Movie/Series',1, df_bored) 
df_bored = ratingInput(301, 'Take/Plan a vacation or travel',1, df_bored) 
df_bored = ratingInput(301, 'Get more sleep/Take a nap',0, df_bored) 
df_bored = ratingInput(301, 'Drink some coffee (moderately)',1, df_bored)
df_bored = ratingInput(301, 'Get Creative: Draw, Paint, Play an instrument',1, df_bored) 
df_bored = ratingInput(301, 'Go shopping',1, df_bored) 
df_bored = ratingInput(301, 'Watch Comedy Movie/Series',1, df_bored) 
df_bored = ratingInput(301, 'Create a schedule/Organize your time/Declutter',0, df_bored) 
df_bored = ratingInput(301, 'Gaming (staying time conscious!)',1, df_bored) 
df_bored = ratingInput(301, 'Watch Action Movie/Series',1, df_bored) 
df_bored = ratingInput(301, 'Reading',1, df_bored) 
df_bored = ratingInput(301, "Write down how you feel/what you're grateful for",0, df_bored)
l2=PredictionsCB(301,df_bored)
l1,RMSE=PredictionsCF(301, df_bored)
print("After Extensive Use:\n",FinalPredictions(l2,l1,df_bored,0.72, 301))






