#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import graphviz

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.tree import DecisionTreeClassifier 
from IPython.display import display, Image
from sklearn.tree import export_graphviz


# In[36]:


train_data = pd.read_csv('data/training_data.csv')
testing_data = pd.read_csv('data/songs_to_classify.csv')


# In[37]:


np.random.seed(100) 


# In[38]:


Xtrain = train_data.copy().drop(columns = ['label'])
Xtrain


# In[39]:


Ytrain = train_data['label']
Ytrain


# In[40]:


#now we are using feature section tool.


# In[41]:


Xtrain1 = Xtrain.copy().drop(columns = ['loudness']) #function was giving error due to negative value I removed laudness from the data but I am consider it for traing my model 
Xtrain1


# In[43]:


best_features = SelectKBest(score_func = chi2, k=9)
fitti = best_features.fit(Xtrain1,Ytrain)
D_score = pd.DataFrame(fitti.scores_)
D_col = pd.DataFrame(Xtrain1.columns)


# In[44]:


feature_scores = pd.concat([D_col,D_score],axis=1)
feature_scores.columns = ['Specs', 'Score'] 


# In[45]:


feature_scores


# In[46]:


#to get the top 10 variables 
print(feature_scores.nlargest(10,'Score'))


# In[47]:


#below is the simple random Forest classifier used 


# In[48]:


model = RandomForestClassifier()
#n_estimator is used to make number of trees . 
model.fit(Xtrain,Ytrain)


# In[49]:


y_test = model.predict(testing_data)
y_test


# In[50]:


scores = cross_val_score(model,Xtrain, Ytrain, cv=10, scoring='accuracy')
print(scores.mean())


# In[53]:


#now removing some of the features which contributed less to the model, our complicated RF classifier 


# In[54]:


Xtrain1 = Xtrain.copy().drop(columns = ['time_signature', 'liveness','mode'])


# In[55]:


model = RandomForestClassifier(n_estimators=100)
#n_estimator is used to make number of trees . 
model.fit(Xtrain1,Ytrain)


# In[56]:


y_test = model.predict(testing_data.copy().drop(columns = ['time_signature', 'liveness','mode']))
y_test


# In[57]:


scores = cross_val_score(model,Xtrain1, Ytrain, cv=10, scoring='accuracy')
print(scores.mean())


# In[58]:


def trimWhiteSpace(ary):
    st=""
    for i in range(len(ary)):
        st+=str(ary[i])
    
    return st;


# In[59]:


trimWhiteSpace(y_test)


# In[60]:


#below function is used to visualize the first tree of our random forest 


# In[61]:


x_grph = train_data.copy().drop(columns = ['label'])


# In[62]:


y_grph = train_data['label']


# In[63]:


model_G = RandomForestClassifier(n_estimators=100)
#n_estimator is used to make number of trees . 
model_G = model_G.fit(x_grph,y_grph)


# In[64]:


model_G.estimators_


# In[65]:


len(model_G.estimators_)


# In[66]:


plt.figure(figsize=(400,300))
tree.plot_tree(model_G.estimators_[1], filled = True)


# In[ ]:





# In[ ]:




