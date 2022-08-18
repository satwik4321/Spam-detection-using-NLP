#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[14]:


data=pd.read_csv(r"C:\Users\sathw\Downloads\smsspamcollection\SMSSpamCollection", sep='\t',names=["label","message"])


# In[15]:


display(data)


# In[16]:


import re
import nltk
nltk.download('stopwords')


# In[20]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps=PorterStemmer()


# In[21]:


corpus=[]
for i in range(0,len(data)):
    review=re.sub('[^a-zA-Z]',' ',data['message'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


# In[53]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[54]:


cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()


# In[55]:


X.shape


# In[56]:


y=pd.get_dummies(data['label'])


# In[57]:


y=y.iloc[:,1].values


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[60]:


from sklearn.naive_bayes import MultinomialNB


# In[69]:


model=MultinomialNB().fit(x_train,y_train)


# In[70]:


y_pred=model.predict(x_test)


# In[63]:


from sklearn.metrics import confusion_matrix


# In[71]:


cm=confusion_matrix(y_test,y_pred)


# In[72]:


print(cm)


# In[73]:


from sklearn.metrics import accuracy_score


# In[74]:


accuracy=accuracy_score(y_test,y_pred)


# In[75]:


print(accuracy)


# In[ ]:




