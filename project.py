#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[29]:


data=pd.read_csv("G:/project_datas/StudentsPerformance.csv")


# In[30]:


data.head()


# In[31]:


data.describe()


# In[32]:


data.info()


# In[33]:


data.shape


# In[34]:


hist = data.hist(bins=5)


# In[35]:


data.corr()


# In[36]:


corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values)


# In[37]:


data.isnull().sum().sum()


# In[38]:


data.isnull().sum()


# In[39]:


data.plot(x='writing score', y='reading score', style='o')  
plt.title('Student Performance')  
plt.xlabel('Writing score')  
plt.ylabel('Reading score')  
plt.show()


# In[40]:


import seaborn as sns
sns.boxplot(x=data['writing score'])


# In[41]:


import seaborn as sns
sns.boxplot(x=data['reading score'])


# In[78]:


x = data.iloc[:, 5:8].values
y = data.iloc[:,1].values


# In[ ]:





# In[79]:


LE=LabelEncoder()
y=LE.fit_transform(y)


# In[ ]:





# In[87]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[94]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=2)
classifier.fit(x_train,y_train)


# In[95]:


y_pred=classifier.predict(x_test)


# In[96]:


score = classifier.score(x_test,y_test)


# In[97]:


score


# In[98]:


###Random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state=0)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[99]:


clf.score(x_test,y_test)


# In[101]:


######Logistic regression 
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression ()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)


# In[102]:


classifier.score(x_test,y_test)


# In[104]:


from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(x_test,y_test)
y_pred = clf.predict(x_test)


# In[105]:


clf.score(x_test,y_test)


# In[ ]:




