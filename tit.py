# Titanic problem on Kaggle

# importing libraries
import csv
import numpy as np
import pandas as pd
import sys
import re as re
from sklearn.svm import SVC

# Importing the dataset
sys.path.append("../home/swapnil/Desktop/titanic/")
train = pd.read_csv('train.csv', error_bad_lines=False)
test = pd.read_csv('test.csv', error_bad_lines=False)
full_data=[train,test]
print(train.info())

# feature engineering

	# 1. Pclass
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


	# 2. Sex
print(train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean())


	# 3. SibSp and Parch
## It stands for Siblings/Spouse and Parents/Children
print(train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean())
## It does not give a very fair idea
print(train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean())
## It also does not give a very good idea
## Combining both of them to form a new feature Family
for data in full_data:
	data['Family']=data['SibSp']+data['Parch']+1
print(train[['Family','Survived']].groupby(['Family'],as_index=False).mean())
## Not a very good feature
## Dividing into alone and not alone category
for data in full_data:
	data['Alone']=0
	data.loc[data['Family']==1,'Alone']=1
print(train[['Alone','Survived']].groupby(['Alone'],as_index=False).mean())


	# 4. Embarked
## Only two missing values so fill them with most common value i.e S
for data in full_data:
	data['Embarked']=data['Embarked'].fillna('S')
print(train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean())


	# 5. Fare
## Some values are 0 in fare so filling them with the median
for data in full_data:
	data['Fare']=data['Fare'].fillna(train['Fare'].median())
## Since Fare has many discrete values therefore categorizing Fare into 4 intervals
train['CatFare'] = pd.qcut(train['Fare'], 4)
print (train[['CatFare', 'Survived']].groupby(['CatFare'], as_index=False).mean())


	# 6. Age
## There are many missing values in Age and thus we replace those with random numbers between age-std and age+std
for data in full_data:
	age_avg=data['Age'].mean()
	age_std=data['Age'].std()
	null_cnt=data['Age'].isnull().sum()
	rand_list=np.random.randint(age_avg-age_std,age_avg+age_std,size=null_cnt)
	data['Age'][np.isnan(data['Age'])]=rand_list
	data['Age']=data['Age'].astype(int)
## Since Fare has many discrete values therefore categorizing Age into 5 intervals
train['CatAge']=pd.qcut(train['Age'],5)
print(train[['CatAge','Survived']].groupby(['CatAge'],as_index=False).mean())

	# 7. Name
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""
for data in full_data:
	data['Title']=data['Name'].apply(get_title)
print(pd.crosstab(train['Title'], train['Sex']))
## Replacing all the small titles into one heading Other
for data in full_data:
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    data['Title']=data['Title'].replace('Mlle', 'Miss')
    data['Title']=data['Title'].replace('Ms', 'Miss')
    data['Title']=data['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


	# 8. Cabin
## A large number of values are missing in this thus Cabin can not be used as a feature


# Data Cleaning
for data in full_data:
	## Mapping Sex
	data['Sex']=data['Sex'].map({'female':0,'male':1}).astype(int)
	## Mapping Titles
	Title_mapping={'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Other':5}
	data['title']=data['Title'].map(Title_mapping).astype(int)
	data['Title']=data['title'].fillna(0)
	## Mapping Embarked
	data['Embarked']=data['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
	## Mapping Age
	data.loc[data['Age']<=19,'Age']=0
    	data.loc[(data['Age']>19) & (data['Age']<=25),'Age']=1
    	data.loc[(data['Age']>25) & (data['Age']<=32),'Age']=2
    	data.loc[(data['Age']>32) & (data['Age']<=40),'Age']=3
    	data.loc[data['Age']>40,'Age']=4
	data['Age']=data['Age'].astype(int)	
	## Mapping fare
	data.loc[data['Fare']<=7.91,'Fare']=0
	data.loc[(data['Fare']>7.91) & (data['Fare']<=14.454),'Fare']=1
	data.loc[(data['Fare']>14.454) & (data['Fare']<=31.0),'Fare']=2
	data.loc[data['Fare']>31.0,'Fare']=3
	data['Fare']=data['Fare'].astype(int)

# Feature Selection
drop_elements = ['PassengerId','Ticket','Parch','Name','SibSp','Cabin','Family']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CatFare','CatAge'], axis = 1)
test  = test.drop(drop_elements, axis = 1)
print (train.head(10))
train = train.values
test  = test.values

# Classifier
## Using Support Vector Machine classifier
clf=SVC(C=5.0,gamma='auto')
clf.fit(train[0:,1:],train[0:,0])
pred=clf.predict(test)
print(pred)
pred=np.reshape(pred,(418,1))
arr=np.zeros([418,1]).astype(int)
for itr in range(arr.size):
	arr[itr]=itr+892
pred=np.append(arr,pred,axis=1)
df=pd.DataFrame(pred)
df.to_csv('Result.csv',header=['PassengerId','Survived'],index=False)
