#!/usr/bin/env python
# coding: utf-8

# # Titanic Classification

# * Import libraries

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# * Load the Titanic dataset

# In[5]:


d = pd.read_csv('titanic.csv')
d


# * Preprocess the data

# In[6]:


d.info()


# In[7]:


d.shape


# In[8]:


#checking for missing values in our dataset
d.isna().sum()


# In[9]:


#checking percentage of missing values from the column overall entries
round(d.isna().sum()/len(d)*100)


# * Handling Missing Values

# In[11]:


#we shall drop cabin column as it has most missing value at 78%
d.drop('Cabin',axis=1,inplace=True)


# In[12]:


#getting passenger titles from the names column
d.Name.head(50)
d['passenger_title'] = d.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[13]:


#As for Age and  Embarked columns we shall replace missing values with the median and  mode  of their column values
d['Age']=d['Age'].fillna(d.Age.median())


# In[14]:


#The mode is the most frequently occurring element in a series.
d['Embarked']=d['Embarked'].fillna(d.Embarked.mode()[0])


# In[15]:


#checking missing values in our dataset
d.isna().sum()


# In[16]:


d.drop('Name',axis=1,inplace=True)


# In[17]:


d.head()


# In[18]:


# Select the features and target variable
features = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch']
target = 'Survived'


# In[19]:


X = d[features]
y = d[target]


# In[20]:


# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)


# In[22]:


X_encoded


# In[21]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[23]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[24]:


# Train the Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[25]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[26]:


y_pred


# In[27]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

