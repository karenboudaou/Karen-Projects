#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:10:33 2022

@author: kaykaydaou
"""
                                                   
#-------#-------#-------#-------#-------: part 1: classification-------#-------#-------#-------
#loading libraries (even those not used)
import pandas as pd
import numpy
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score


#Importing data
kickstarter2=pd.read_excel("/Users/kaykaydaou/Desktop/MCGILL U3/FALL 22/INSY 446 - data mining for business analytics/Individual project due 4 dec/Kickstarter.xlsx")

# pre=processing
# duplicated_id=kickstarter1['project_id'].duplicated()==True #checking if there is duplicated data
# duplicated_id.sum() #result=0 so there are no duplicated projects
kickstarter2=kickstarter2.drop(columns=['name']) #dropping name fields => they have a different value for each record so they are not useful in our data mining algorithm
kickstarter2=kickstarter2.set_index('project_id') #setting the id as index because we wont be using it but it's to refer to the project
kickstarter2.isna().sum() #visualize which columns have null values
#category has 1684 and launch_to_state_change_days has 13182 => drop these 2 (and anyway we dont care about the state because we're assuming we dont know the status of the project)
#category seems like an important variable => not dropping it but rather dropping missing values (because when i dropped it i had lower accuracy scores)
kickstarter2=kickstarter2.drop(columns=['launch_to_state_change_days']) 
kickstarter2.isna().sum()  #category, name_len, name_len_clean, blurb_len, blurb_len_clean to drop the rows of these missing values in the columns
kickstarter2=kickstarter2.dropna() 
                     
# Only keeping failed and successful goals
kickstarter2 = kickstarter2[(kickstarter2.state == "failed") | (kickstarter2.state == "successful")]
# kickstarter2["state"].value_counts()##there is way more failed than successful

#converting goal to usd currency
kickstarter2['goal_usd'] = kickstarter2.goal * kickstarter2.static_usd_rate
#dropping goal and static usd rate because no longer needed
kickstarter2 = kickstarter2.drop(columns=['goal', 'static_usd_rate'])
#normalizing - commented it because i want to standardize everything later
# kickstarter2['goal_usd'] = (kickstarter2['goal_usd'] - kickstarter2['goal_usd'].mean()) / kickstarter2['goal_usd'].std() 

#creating a new column of create to deadline
kickstarter2['create_to_deadline'] = (kickstarter2['create_to_launch_days'] + kickstarter2['launch_to_deadline_days'])
#dropping goal and static usd rate because no longer needed
kickstarter2 = kickstarter2.drop(columns=['create_to_launch_days', 'launch_to_deadline_days'])

##visualizing the data
kickstarter2['country'].value_counts() #us seams to be submitting the most campaigns followed by GB then CA
kickstarter2['created_at_yr'].value_counts()
kickstarter2['created_at_weekday'].value_counts()
kickstarter2['deadline_weekday'].value_counts()
kickstarter2['category'].value_counts()

#####first approach tested: grouping countries by continents
# kickstarter2['America']=((kickstarter2.country=='US')+
#                         (kickstarter2.country=='CA')+
#                         (kickstarter2.country=='MX'))
# kickstarter2['America']=kickstarter2['America'].astype(int) #true = 1 and false = 0

# kickstarter2['Europe']=((kickstarter2.country=='GB')+
#                         (kickstarter2.country=='DE')+
#                         (kickstarter2.country=='NL')+
#                         (kickstarter2.country=='FR')+
#                         (kickstarter2.country=='IT')+
#                         (kickstarter2.country=='ES')+
#                         (kickstarter2.country=='DK')+
#                         (kickstarter2.country=='SE')+
#                         (kickstarter2.country=='CH')+
#                         (kickstarter2.country=='IE')+
#                         (kickstarter2.country=='NO')+
#                         (kickstarter2.country=='AT')+
#                         (kickstarter2.country=='BE')+
#                         (kickstarter2.country=='LU'))
# kickstarter2['Europe']=kickstarter2['Europe'].astype(int)
                                                
# kickstarter2['Oceania']=((kickstarter2.country=='AU')+
#                         (kickstarter2.country=='NZ'))
# kickstarter2['Oceania']=kickstarter2['Oceania'].astype(int)
                        
# kickstarter2['Asia']=((kickstarter2.country=='HK')+
#                         (kickstarter2.country=='SG'))
# kickstarter2['Asia']=kickstarter2['Asia'].astype(int)

# #another way of getting true and false
# #kickstarter[['America', 'Europe']]=kickstarter[['America', 'Europe']].astype(int)
# #can drop country column now
# kickstarter2=kickstarter2.drop(columns='country')

# #grouping creation by weekday and weekend: created_at_weekday
# kickstarter2['created_on_a_weekday']=((kickstarter2.created_at_weekday=='Monday')+
#                                       (kickstarter2.created_at_weekday=='Tuesday')+
#                                       (kickstarter2.created_at_weekday=='Wednesday')+
#                                       (kickstarter2.created_at_weekday=='Thursday')+
#                                       (kickstarter2.created_at_weekday=='Friday'))
# kickstarter2['created_on_a_weekday']=kickstarter2['created_on_a_weekday'].astype(int) #true = 1 and false = 0


# kickstarter2['created_on_a_weekend']=((kickstarter2.created_at_weekday=='Saturday')+
#                                       (kickstarter2.created_at_weekday=='Sunday'))
# kickstarter2['created_on_a_weekend']=kickstarter2['created_on_a_weekend'].astype(int) #true = 1 and false = 0

# #can drop created_at_weekday column now
# kickstarter2=kickstarter2.drop(columns='created_at_weekday')
                                     
# #grouping deadline by weekday and weekend: deadline_weekday
# kickstarter2['deadline_on_a_weekday']=((kickstarter2.deadline_weekday=='Monday')+
#                                       (kickstarter2.deadline_weekday=='Tuesday')+
#                                       (kickstarter2.deadline_weekday=='Wednesday')+
#                                       (kickstarter2.deadline_weekday=='Thursday')+
#                                       (kickstarter2.deadline_weekday=='Friday'))
# kickstarter2['deadline_on_a_weekday']=kickstarter2['deadline_on_a_weekday'].astype(int) #true = 1 and false = 0


# kickstarter2['deadline_on_a_weekend']=((kickstarter2.deadline_weekday=='Saturday')+
#                                       (kickstarter2.deadline_weekday=='Sunday'))
# kickstarter2['deadline_on_a_weekend']=kickstarter2['deadline_on_a_weekend'].astype(int) #true = 1 and false = 0

# #can drop deadline_weekday column now
# kickstarter2=kickstarter2.drop(columns='deadline_weekday')

# #grouping category by bigger groups
# # kickstarter2['Genres']=((kickstarter2.category=='Comedy') + (kickstarter2.category=='Thrillers')+
# #                         (kickstarter2.category=='Musical'))

# kickstarter2['Technology']=((kickstarter2.category=='Hardware') + (kickstarter2.category=='Web')+
#                         (kickstarter2.category=='Software')+(kickstarter2.category=='Gadgets')+
#                         (kickstarter2.category=='Apps')+ (kickstarter2.category=='Wearables')+
#                         (kickstarter2.category=='Sound')+(kickstarter2.category=='Robots')+
#                         (kickstarter2.category=='Flight')+(kickstarter2.category=='Makerspaces'))
# kickstarter2['Technology']=kickstarter2['Technology'].astype(int)

# kickstarter2['Theater']=((kickstarter2.category=='Plays') + (kickstarter2.category=='Musical')+
#                         (kickstarter2.category=='Theater')+ (kickstarter2.category=='Experimental')+
#                         (kickstarter2.category=='Immersive')+ (kickstarter2.category=='Spaces')+
#                         (kickstarter2.category=='Comedy'))
# kickstarter2['Theater']=kickstarter2['Theater'].astype(int)

                        
# kickstarter2['Film']=((kickstarter2.category=='Shorts')+(kickstarter2.category=='Thrillers')+
#                       (kickstarter2.category=='Experimental')+(kickstarter2.category=='Thrillers')+
#                       (kickstarter2.category=='Webseries')+(kickstarter2.category=='Comedy'))
# kickstarter2['Film']=kickstarter2['Film'].astype(int)


# kickstarter2['Photography_Music_Academic']=((kickstarter2.category=='Places')+
#                                             (kickstarter2.category=='Blues')+
#                                             (kickstarter2.category=='Academic'))
# kickstarter2['Photography_Music_Academic']=kickstarter2['Photography_Music_Academic'].astype(int)

# #can drop category column now
# kickstarter2=kickstarter2.drop(columns='category')                      

# #grouping year by year of creation before 2013 and year of creation after 2013
# kickstarter2['created_bef_2013']=((kickstarter2.created_at_yr==int('2009'))+
#                                   (kickstarter2.created_at_yr==int('2010'))+
#                                   (kickstarter2.created_at_yr==int('2011'))+
#                                   (kickstarter2.created_at_yr==int('2012'))+
#                                   (kickstarter2.created_at_yr==int('2013')))

# kickstarter2['created_bef_2013']=kickstarter2['created_bef_2013'].astype(int)

# kickstarter2['created_aft_2013']=((kickstarter2.created_at_yr==int('2014'))+
#                                   (kickstarter2.created_at_yr==int('2015'))+
#                                   (kickstarter2.created_at_yr==int('2016'))+
#                                   (kickstarter2.created_at_yr==int('2017')))

# kickstarter2['created_aft_2013']=kickstarter2['created_aft_2013'].astype(int)

# #can drop created_at_yr column now
# kickstarter2=kickstarter2.drop(columns='created_at_yr')  

#####second approach: dummify
#i noticed my accuracy scores are better when i dummify my categorical variables instead of grouping them
#hence i will be using X with my grouped variables for the clustering part of the assignment as it is better to interpret

#dummifying instead of grouping 
kickstarter2=pd.get_dummies(kickstarter2,columns=['category','country','created_at_weekday','deadline_weekday','created_at_yr'])

#setting target variables
y = kickstarter2[['state']]
mapping = {"failed": 0, "successful": 1} 
y = y.replace({"state":mapping})

#X variable contains: created_at_yr,created_at_weekday,deadline_weekday,category,country, name_len_clean, blurb_len_clean,create_to_deadline,goal_usd

df1_part1 = kickstarter2[['name_len_clean','blurb_len_clean','create_to_deadline','goal_usd']]
df2_part1= kickstarter2.iloc[:,35:]
X=pd.concat([df1_part1,df2_part1],axis=1)

##standardization: important to standardize using minmax (between 0 and 1)
X_std=X.copy()
X_std[['goal_usd','name_len_clean','blurb_len_clean', 'create_to_deadline'
       ]] = MinMaxScaler().fit_transform(X_std[['goal_usd','name_len_clean',
                                'blurb_len_clean', 'create_to_deadline']])

#---------------------------------------model1: logistic regression---------------------------------------
#                                         without standardization
#Split the data
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.33, random_state=4)

# build and run the model
lr=LogisticRegression()
model=lr.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred=model.predict(X_test)

# Calculate the accuracy score
print(metrics.accuracy_score(y_test,y_test_pred)) #result = 0.6674483052654019

#                                        with standardization
#Split the data
X_train, X_test, y_train, y_test= train_test_split(X_std,y,test_size=0.33, random_state=4)

# build and run the model
lr1=LogisticRegression() #add max iter = 1000
model1=lr1.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred=model1.predict(X_test)

# Calculate the accuracy score
print(metrics.accuracy_score(y_test,y_test_pred)) #result = 0.7145597953528032

#-----------------------------------------model2: KNN------------------------------------------------------ 
#Split the data
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.33, random_state=4)

# Run K-NN
knn = KNeighborsClassifier(n_neighbors=8) #randomly picked 8
model2 = knn.fit(X_train, y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model2.predict(X_test)

# Calculate the accuracy score
print(metrics.accuracy_score(y_test,y_test_pred)) #result = 0.68684715412492

#second approach: choosing the optimal k
for i in range (1,11):
    knn2=KNeighborsClassifier(n_neighbors=i)
    model2_2=knn2.fit(X_train,y_train)
    y_test_pred=model2_2.predict(X_test)
    print(accuracy_score(y_test,y_test_pred))
    
    #found optimal k to be 9 with highest accuracy score
    #result = 0.6872735024515029

#---------------------------------------model3: Decision tree--------------------------------------------
#finding optimal max_depth:
for i in range (2,21):
    model3_3=DecisionTreeClassifier(max_depth=i, random_state=4)
    scores=cross_val_score(estimator=model3_3,X=X,y=y,cv=5)
    print(i,':',numpy.average(scores))

#optimal max depth = 8 with result of 0.7227387142711098

#                               train test split without standardization
# Separate the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

# Run decision tree 
decisiontree=DecisionTreeClassifier(max_depth=4) #randomly picked 4
model3=decisiontree.fit(X_train, y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model3.predict(X_test)

# Calculate the accuracy score
print(metrics.accuracy_score(y_test,y_test_pred))  #result = 0.7147729695160947


#                                        with standardization
# Separate the data
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.33, random_state=4)

# Run decision tree 
decisiontree=DecisionTreeClassifier(max_depth=4)
model4=decisiontree.fit(X_train, y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model4.predict(X_test)

# calculate the accuracy score
print(metrics.accuracy_score(y_test,y_test_pred)) #result = 0.7149861436793861 with max_depth =4

#changed max_depth to 8 and found 0.7192496269452142 => slightly better

#---------------------------------------model4: Random Forest-----------------------------------------
#varying max features
for i in range(2,7):
    model5_5=RandomForestClassifier(random_state=4, max_features=i, n_estimators=100)
    scores=cross_val_score(estimator=model5_5,X=X,y=y,cv=5)
    print(i,':',numpy.average(scores))

#found optimal max_feature to be 6 with result = 0.7368793013099572    

#                           basic random forest without standardization

# Separate the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)

# run random forest
randomforest=RandomForestClassifier()
model5=randomforest.fit(X_train,y_train)

#using the model to predict the results based on the test dataset
y_test_pred=model5.predict(X_test)

# calculate the accuracy score
accuracy_score(y_test,y_test_pred)     #result = 0.7463227456832232

# #adding hyperparameters
# # run random forest
# randomforest2=RandomForestClassifier(max_features=6, n_estimators=100)
# model5_2=randomforest.fit(X_train,y_train)

# #using the model to predict the results based on the test dataset
# y_test_pred=model5_2.predict(X_test)

# # calculate the accuracy score
# accuracy_score(y_test,y_test_pred)  #result = 0.7399275207844809 (my result is better without hyperparameters)

#                                        with standardization
# Separate the data
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.33, random_state=4)

# run random forest
randomforest3=RandomForestClassifier()
model6=randomforest3.fit(X_train,y_train)

#using the model to predict the results based on the test dataset
y_test_pred=model6.predict(X_test)

#calculate the accuracy score
accuracy_score(y_test,y_test_pred)     #result = 0.7478149648262631 #best result of all models

#---------------------------------model5: GBT: MODEL I AM CHOOSING---------------------------------------------------------
#varying min_samples split
for i in range (2,10):
    model7_1=GradientBoostingClassifier(random_state=4,min_samples_split=i,n_estimators=100)
    scores=cross_val_score(estimator=model7_1,X=X,y=y,cv=5)
    print(i,':',numpy.average(scores))
    
    #best score is 4 with result = 0.7427888986443486

#                                        without standardization
# Separate the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=4)

# run GBT
gbt=GradientBoostingClassifier()
model7=gbt.fit(X_train,y_train)

#using the model to predict the results based on the test dataset
y_test_pred=model7.predict(X_test)

#calculate the accuracy score
accuracy_score(y_test,y_test_pred)    #result = 0.7465359198465146   #second best result

#                                        with standardization
# Separate the data
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.33,random_state=4)

# run GBT
gbt=GradientBoostingClassifier()
model8=gbt.fit(X_train,y_train)

#using the model to predict the results based on the test dataset
y_test_pred=model8.predict(X_test)

#calculate the accuracy score
accuracy_score(y_test,y_test_pred)    #result = 0.7450437007034747

#------------------------------------model6: ANN------------------------------------------------------
# Separate the data
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.33, random_state=4)

# run MLP
mlp=MLPClassifier(hidden_layer_sizes=(11),max_iter=1000,random_state=4)
model9=mlp.fit(X_train,y_train)

#using the model to predict the results based on the test dataset
y_test_pred=model9.predict(X_test)

#calculate the accuracy score
accuracy_score(y_test, y_test_pred)   #result = 0.7049669580046898

##decided not to vary the number of hidden layers because the number of hidden layers doesn’t matter when u have limited observations
#it matters when i have billions of observations


#-------#-------#-------#-------#-------part 2: clustering-------#-------#-------#-------#-------
#grouping is better for the clustering and get dummies is better for the score
#resetting my X variable with more concise data by grouping for more efficient clustering interpretation

kickstarter3=pd.read_excel("/Users/kaykaydaou/Desktop/MCGILL U3/FALL 22/INSY 446 - data mining for business analytics/Individual project due 4 dec/Kickstarter.xlsx")

# pre=processing
kickstarter3=kickstarter3.drop(columns=['name']) 
kickstarter3=kickstarter3.set_index('project_id') 
kickstarter3.isna().sum() 
kickstarter3=kickstarter3.drop(columns=['launch_to_state_change_days']) 
kickstarter3=kickstarter3.dropna() 
                     
# Only keeping failed and successful goals
kickstarter3 = kickstarter3[(kickstarter3.state == "failed") | (kickstarter3.state == "successful")]

#converting goal to usd currency
kickstarter3['goal_usd'] = kickstarter3.goal * kickstarter3.static_usd_rate
#dropping goal and static usd rate because no longer needed
kickstarter3 = kickstarter3.drop(columns=['goal', 'static_usd_rate'])
# kickstarter3['goal_usd'] = (kickstarter3['goal_usd'] - kickstarter3['goal_usd'].mean()) / kickstarter3['goal_usd'].std() 

#feature engineering
#creating a new column of create to deadline
kickstarter3['create_to_deadline'] = (kickstarter3['create_to_launch_days'] + kickstarter3['launch_to_deadline_days'])
#dropping goal and static usd rate because no longer needed
kickstarter3 = kickstarter3.drop(columns=['create_to_launch_days', 'launch_to_deadline_days'])

#grouping countries by continents
kickstarter3['America']=((kickstarter3.country=='US')+
                        (kickstarter3.country=='CA')+
                        (kickstarter3.country=='MX'))
kickstarter3['America']=kickstarter3['America'].astype(int) #true = 1 and false = 0

kickstarter3['Europe']=((kickstarter3.country=='GB')+
                        (kickstarter3.country=='DE')+
                        (kickstarter3.country=='NL')+
                        (kickstarter3.country=='FR')+
                        (kickstarter3.country=='IT')+
                        (kickstarter3.country=='ES')+
                        (kickstarter3.country=='DK')+
                        (kickstarter3.country=='SE')+
                        (kickstarter3.country=='CH')+
                        (kickstarter3.country=='IE')+
                        (kickstarter3.country=='NO')+
                        (kickstarter3.country=='AT')+
                        (kickstarter3.country=='BE')+
                        (kickstarter3.country=='LU'))
kickstarter3['Europe']=kickstarter3['Europe'].astype(int)
                                                
kickstarter3['Oceania']=((kickstarter3.country=='AU')+
                        (kickstarter3.country=='NZ'))
kickstarter3['Oceania']=kickstarter3['Oceania'].astype(int)
                        
kickstarter3['Asia']=((kickstarter3.country=='HK')+
                        (kickstarter3.country=='SG'))
kickstarter3['Asia']=kickstarter3['Asia'].astype(int)

#can drop country column now
kickstarter3=kickstarter3.drop(columns='country')

#grouping creation by weekday and weekend: created_at_weekday
kickstarter3['created_on_a_weekday']=((kickstarter3.created_at_weekday=='Monday')+
                                      (kickstarter3.created_at_weekday=='Tuesday')+
                                      (kickstarter3.created_at_weekday=='Wednesday')+
                                      (kickstarter3.created_at_weekday=='Thursday')+
                                      (kickstarter3.created_at_weekday=='Friday'))
kickstarter3['created_on_a_weekday']=kickstarter3['created_on_a_weekday'].astype(int) #true = 1 and false = 0


kickstarter3['created_on_a_weekend']=((kickstarter3.created_at_weekday=='Saturday')+
                                      (kickstarter3.created_at_weekday=='Sunday'))
kickstarter3['created_on_a_weekend']=kickstarter3['created_on_a_weekend'].astype(int) #true = 1 and false = 0

#can drop created_at_weekday column now
kickstarter3=kickstarter3.drop(columns='created_at_weekday')
                                     
#grouping deadline by weekday and weekend: deadline_weekday
kickstarter3['deadline_on_a_weekday']=((kickstarter3.deadline_weekday=='Monday')+
                                      (kickstarter3.deadline_weekday=='Tuesday')+
                                      (kickstarter3.deadline_weekday=='Wednesday')+
                                      (kickstarter3.deadline_weekday=='Thursday')+
                                      (kickstarter3.deadline_weekday=='Friday'))
kickstarter3['deadline_on_a_weekday']=kickstarter3['deadline_on_a_weekday'].astype(int) #true = 1 and false = 0


kickstarter3['deadline_on_a_weekend']=((kickstarter3.deadline_weekday=='Saturday')+
                                      (kickstarter3.deadline_weekday=='Sunday'))
kickstarter3['deadline_on_a_weekend']=kickstarter3['deadline_on_a_weekend'].astype(int) #true = 1 and false = 0

#can drop deadline_weekday column now
kickstarter3=kickstarter3.drop(columns='deadline_weekday')

#grouping category by bigger groups
kickstarter3['Technology']=((kickstarter3.category=='Hardware') + (kickstarter3.category=='Web')+
                        (kickstarter3.category=='Software')+(kickstarter3.category=='Gadgets')+
                        (kickstarter3.category=='Apps')+ (kickstarter3.category=='Wearables')+
                        (kickstarter3.category=='Sound')+(kickstarter3.category=='Robots')+
                        (kickstarter3.category=='Flight')+(kickstarter3.category=='Makerspaces'))
kickstarter3['Technology']=kickstarter3['Technology'].astype(int)

kickstarter3['Theater']=((kickstarter3.category=='Plays') + (kickstarter3.category=='Musical')+
                        (kickstarter3.category=='Theater')+ (kickstarter3.category=='Experimental')+
                        (kickstarter3.category=='Immersive')+ (kickstarter3.category=='Spaces')+
                        (kickstarter3.category=='Comedy'))
kickstarter3['Theater']=kickstarter3['Theater'].astype(int)

                        
kickstarter3['Film']=((kickstarter3.category=='Shorts')+(kickstarter3.category=='Thrillers')+
                      (kickstarter3.category=='Experimental')+(kickstarter3.category=='Thrillers')+
                      (kickstarter3.category=='Webseries')+(kickstarter3.category=='Comedy'))
kickstarter3['Film']=kickstarter3['Film'].astype(int)


kickstarter3['Photography_Music_Academic']=((kickstarter3.category=='Places')+
                                            (kickstarter3.category=='Blues')+
                                            (kickstarter3.category=='Academic'))
kickstarter3['Photography_Music_Academic']=kickstarter3['Photography_Music_Academic'].astype(int)

#can drop category column now
kickstarter3=kickstarter3.drop(columns='category')                      

#grouping year by year of creation before 2013 and year of creation after 2013
kickstarter3['created_bef_2013']=((kickstarter3.created_at_yr==int('2009'))+
                                  (kickstarter3.created_at_yr==int('2010'))+
                                  (kickstarter3.created_at_yr==int('2011'))+
                                  (kickstarter3.created_at_yr==int('2012'))+
                                  (kickstarter3.created_at_yr==int('2013')))

kickstarter3['created_bef_2013']=kickstarter3['created_bef_2013'].astype(int)

kickstarter3['created_aft_2013']=((kickstarter3.created_at_yr==int('2014'))+
                                  (kickstarter3.created_at_yr==int('2015'))+
                                  (kickstarter3.created_at_yr==int('2016'))+
                                  (kickstarter3.created_at_yr==int('2017')))

kickstarter3['created_aft_2013']=kickstarter3['created_aft_2013'].astype(int)

#can drop created_at_yr column now
kickstarter3=kickstarter3.drop(columns='created_at_yr')  

#X variable contains continents, created_on_a_weekday, created_on_a_weekend, deadline_on_a_weekday, deadline_on_a_weekend
# Technology, Theater, Film, Photography_Music_Academic,created_bef_2013,created_aft_2013
# name_len_clean, blurb_len_clean, create_to_deadline, goal_usd

#converting failed and successful to 0 and 1
# mapping2 = {"failed": 0, "successful": 1} 
# kickstarter3 = kickstarter3.replace({"state":mapping2})

#dummifying state
kickstarter3=pd.get_dummies(kickstarter3,columns=['state'])

df1_pt2 = kickstarter3[['name_len_clean','blurb_len_clean','create_to_deadline','goal_usd']]
df2_pt2= kickstarter3.iloc[:,34:]
X_pt2=pd.concat([df1_pt2,df2_pt2],axis=1)

##standardization: important to standardize using minmax (between 0 and 1)
X_std_pt2=X_pt2.copy()
X_std_pt2[['goal_usd','name_len_clean','blurb_len_clean',
       'create_to_deadline']] = MinMaxScaler().fit_transform(X_std_pt2[['goal_usd',
            'name_len_clean','blurb_len_clean', 'create_to_deadline']])

#performing pca to reduce the data
pca_pt2=PCA(n_components=10)
pca_pt2.fit(X_pt2)
X_new=pca_pt2.transform(X_pt2)
  
#elbow method to find optimal k                                                                      
withinss=[]
for i in range (1,10):
    kmeans2=KMeans(n_clusters=i)
    model10=kmeans2.fit(X_new)
    withinss.append(model10.inertia_)
    
pyplot.plot([1,2,3,4,5,6,7,8,9],withinss)                                                                         
                                         
#clustering using kmeans                                                                                                  
kmeans_pt2=KMeans(n_clusters=4)
model11=kmeans_pt2.fit(X_std_pt2)
labels_pt2=kmeans_pt2.predict(X_std_pt2)

centroids1=model11.cluster_centers_

X_pt2['ClusterMembership']=labels_pt2

cluster_1_pt2=X_pt2[X_pt2['ClusterMembership']==0].mean()
cluster_2_pt2=X_pt2[X_pt2['ClusterMembership']==1].mean()
cluster_3_pt2=X_pt2[X_pt2['ClusterMembership']==2].mean()
cluster_4_pt2=X_pt2[X_pt2['ClusterMembership']==3].mean()

clusters_pt2=pd.concat([cluster_1_pt2,cluster_2_pt2,cluster_3_pt2,cluster_4_pt2],axis=1)
clusters_pt2   

#clustering usking X_new
# kmeans_pt3=KMeans(n_clusters=4)
# model12=kmeans_pt3.fit(X_new)
# labels_pt2_2=kmeans_pt3.predict(X_new)

# centroids2=model12.cluster_centers_

# X_pt2['ClusterMembership']=labels_pt2_2

# cluster_1_pt2_2=X_pt2[X_pt2['ClusterMembership']==0].mean()
# cluster_2_pt2_2=X_pt2[X_pt2['ClusterMembership']==1].mean()
# cluster_3_pt2_2=X_pt2[X_pt2['ClusterMembership']==2].mean()
# cluster_4_pt2_2=X_pt2[X_pt2['ClusterMembership']==3].mean()

# clusters_pt2_2=pd.concat([cluster_1_pt2_2,cluster_2_pt2_2,cluster_3_pt2_2,cluster_4_pt2_2],axis=1)
# clusters_pt2_2   
# results are better using X_Std => not doing this method                                                             
                                                                        
#performing elbow method without pca
# from sklearn.cluster import KMeans
# withinsss=[]
# for i in range (1,10):
#     kmeans3=KMeans(n_clusters=i)
#     model12=kmeans2.fit(X_std_pt2)
#     withinsss.append(model12.inertia_)
    
# from matplotlib import pyplot
# pyplot.plot([1,2,3,4,5,6,7,8,9],withinsss)
# terrible results : dropping it    


# #optimal k using silhouette method => did not use it because i got very high optimal k
# from sklearn.metrics import silhouette_score
# for i in range (2,8):    
#     kmeans_pt2_2 = KMeans(n_clusters=i)
#     model13 = kmeans_pt2_2.fit(X_std_pt2)
#     labels_pt2_2 = model13.labels_
#     print(i,':',silhouette_score(X_std_pt2,labels_pt2_2))


#-------#-------#-------#-------GRADING-------#-------#-------#-------#-------
                                                      
                                                                                                                                       
#Importing grading data
kickstarter=pd.read_excel("/Users/kaykaydaou/Desktop/MCGILL U3/FALL 22/INSY 446 - data mining for business analytics/Individual project due 4 dec/Kickstarter-Grading-Sample.xlsx")

# pre=processing grading data
kickstarter=kickstarter.drop(columns=['name']) #dropping name fields => they have a different value for each record so they are not useful in our data mining algorithm
kickstarter=kickstarter.set_index('project_id') #setting the id as index because we wont be using it but it's to refer to the project
kickstarter.isna().sum() 
kickstarter=kickstarter.drop(columns=['launch_to_state_change_days']) 
kickstarter=kickstarter.dropna() 
                     
# Only keeping failed and successful goals
kickstarter = kickstarter[(kickstarter.state == "failed") | (kickstarter.state == "successful")]

#converting goal to usd currency
kickstarter['goal_usd'] = kickstarter.goal * kickstarter.static_usd_rate
#dropping goal and static usd rate because no longer needed
kickstarter = kickstarter.drop(columns=['goal', 'static_usd_rate'])

#creating a new column of create to deadline
kickstarter['create_to_deadline'] = (kickstarter['create_to_launch_days'] + kickstarter['launch_to_deadline_days'])
#dropping goal and static usd rate because no longer needed
kickstarter = kickstarter.drop(columns=['create_to_launch_days', 'launch_to_deadline_days'])

#dummifying categorical variables
kickstarter=pd.get_dummies(kickstarter,columns=['category','country','created_at_weekday','deadline_weekday','created_at_yr'])

#Setup the variables
y_grading = kickstarter[['state']]
mapping = {"failed": 0, "successful": 1} 
y_grading = y_grading.replace({"state":mapping})

df1= kickstarter[['name_len_clean','blurb_len_clean','create_to_deadline','goal_usd']]
df2= kickstarter.iloc[:,35:]
X_grading=pd.concat([df1,df2],axis=1)

##standardization: important to standardize using minmax (between 0 and 1)
X_std_final=X_grading.copy()
X_std_final[['goal_usd','name_len_clean','blurb_len_clean', 'create_to_deadline'
       ]] = MinMaxScaler().fit_transform(X_std_final[['goal_usd','name_len_clean',
                                'blurb_len_clean', 'create_to_deadline']])
                                                      
# GBT model - using my best model (model 7)

# Apply the model previously trained to the grading data
y_grading_pred = model7.predict(X_grading)

# Calculate the accuracy score
accuracy_score(y_grading, y_grading_pred)




