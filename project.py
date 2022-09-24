#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[2]:


#data collection
df=pd.read_csv('C:/Users/varsh/Downloads/diabetes (1).csv')


# In[3]:


#printing the first 5 rows of the dataset
df.head()


# In[4]:


df.shape


# In[5]:


#correlation in my features
import seaborn as sns
corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
#heatmap
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn_r")


# In[6]:


df.corr


# In[7]:


#getting some statictical information of the dataframe
df.describe()


# In[8]:


df['Outcome'].value_counts()


# In[9]:


df.groupby('Outcome').mean()


# In[10]:


#sepearating data and variable
x=df.drop(columns ='Outcome',axis=1)
y=df['Outcome']


# In[11]:


x


# In[14]:


y


# In[15]:


#data standardizations
scaler=StandardScaler()


# In[16]:


scaler.fit(x)


# In[17]:


#transform this data
standardized_data=scaler.transform(x)


# In[18]:


standardized_data


# In[19]:


x=standardized_data
y=df['Outcome']


# In[20]:


x


# In[21]:


y


# In[22]:


#train and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[23]:


x.shape,x_train.shape,x_test.shape


# In[24]:


#training the model
classifier=svm.SVC(kernel='linear')


# In[25]:


#training the svm classifier
classifier.fit(x_train,y_train)


# In[26]:


#model evaluation
x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[27]:


"accuracy score of train data",training_data_accuracy


# In[28]:


#accuract score on the test data
x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
"accuracy score of test data",test_data_accuracy


# In[29]:


#making the prediction

input_data=(5,117,92,0,0,34.1,0.337,38)
# to numoy array
input_data_as_array=np.asarray(input_data)
#reshape the array for one instance
input_data_reshaped=input_data_as_array.reshape(1,-1)
#standaridize the input data
std_data=scaler.transform(input_data_reshaped)


#prediction

prediction=classifier.predict(std_data)


# In[30]:


std_data


# In[31]:


prediction


# In[32]:


if(prediction[0]==0):
    print("According to given set of data we can predict that this person is not going to be diabetic")
else:
    print("According to given set of data we can predict that this person is going to be diabetic")

