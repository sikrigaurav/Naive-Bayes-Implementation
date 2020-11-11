#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
from functools import reduce


# In[ ]:


#################################################################
#Relevent Functions
#################################################################


# In[2]:


#Function to calculate unique number of documents in a given dataframe
def unique_doc(data):
    return len(list(set(data["doc"])))
    


# In[3]:


#Function to calculate words in each class
def count(df,word,class_1,class_2):
    
    d=df[df["word"]==word] 
    d1=d[d["class"]==1]
    d2=d[d["class"]==2]

    class_1[0].append(word)
    class_1[1].append(unique_doc(d1))
    class_2[0].append(word)
    class_2[1].append(unique_doc(d2))

    return class_1,class_2
    
    


# In[4]:


#Function for single data point (single docID) classification:
def prediction(dataframe,nwords,class1_dict,class2_dict,prob1,prob2):
    joint_prob_class1=[]
    joint_prob_class2=[]
    
    for i in list(range(1,nwords+1)):
        if i in list(dataframe["word"]):
            joint_prob_class1.append(class1_dict[i])
        else:
            joint_prob_class1.append(1-class1_dict[i])
    
    for j in list(range(1,nwords+1)):
        if j in list(dataframe["word"]):
            joint_prob_class2.append(class2_dict[j])
        else:
            joint_prob_class2.append(1-class2_dict[j])
        
    class1_join_prob=(reduce((lambda x, y: x * y), joint_prob_class1))    
    class2_join_prob=(reduce((lambda x, y: x * y), joint_prob_class2))
    
    if class1_join_prob*prob1>class2_join_prob*prob2:
        return 1
    else:
        return 2



# In[5]:


#Naive Bayes classifier
def nb_classifier(data,nwords,class1_dict,class2_dict,prob1,prob2):
    class_pred=[]
    for i in list(range(1,max(data["doc"]+1))):
        if i in list(data["doc"]):
            class_pred.append(prediction(data[data["doc"]==i],nwords,class1_dict,class2_dict,prob1,prob2))
    return class_pred
    
    


# In[6]:


#Function to calculate accuracy
def accuracy(data,actual_y,pred_y):
    accuracy=[]
    relevent_labels=[]
    no_doc=unique_doc(data[["doc"]])
    
    for i in list(set(data["doc"])):
        relevent_labels.append(list(actual_y)[i-1])

    for i in range(len(pred_y)):
        if pred_y[i]==relevent_labels[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    
    return sum(accuracy)/no_doc
    #return relevent_labels


# In[ ]:


#################################################################
#Data Preprocessing
#################################################################


# In[7]:


#Loading Data
xtrain = np.loadtxt(fname = "C:/Maruf/UWaterloo/Winter2020/CS 686/A3-AI/Data/trainData.txt")
xtrain=xtrain.astype(int)
ytrain = np.loadtxt(fname = "C:/Maruf/UWaterloo/Winter2020/CS 686/A3-AI/Data/trainLabel.txt")

xtest = np.loadtxt(fname = "C:/Maruf/UWaterloo/Winter2020/CS 686/A3-AI/Data/testData.txt")
xtest=xtest.astype(int)
ytest = np.loadtxt(fname = "C:/Maruf/UWaterloo/Winter2020/CS 686/A3-AI/Data/testLabel.txt")

#ytrain=np.reshape(ytrain,(1500,1))
#ytest=np.reshape(ytest,(1500,1))

words=[line.rstrip('\n') for line in open("C:/Maruf/UWaterloo/Winter2020/CS 686/A3-AI/Data/words.txt")]

nwords=max(max(xtrain[:,1]),max(xtest[:,1]))
nwords=nwords.astype(int)
nobs=max(max(xtrain[:,0]),max(xtest[:,0]))
nobs=nobs.astype(int)


# In[8]:


#converting training data related arrays to a single training dataframe
train_df=pd.concat([pd.DataFrame(xtrain[:,0],columns=["doc"]), pd.DataFrame(xtrain[:,1],columns=["word"]),pd.DataFrame([1]*xtrain.shape[0],columns=["class"])],axis=1)


# In[9]:


#working with the class column in training dataframe
for i in range(len(train_df["doc"])):
    if train_df["doc"][i]>750:
        train_df["class"][i]=2


# In[10]:


#converting test data related arrays to a single test dataframe
test_df=pd.concat([pd.DataFrame(xtest[:,0],columns=["doc"]), pd.DataFrame(xtest[:,1],columns=["word"])],axis=1)


# In[ ]:


#################################################################
#Calculations
#################################################################


# In[30]:


#Count of words in each class
class_1=[[],[]]
class_2=[[],[]]
for i in range(1,nwords+1):
    class_1,class_2=count(train_df,i,class_1,class_2)


# In[107]:


#Count of classes/labels
no_class1=unique_doc(train_df[["doc","class"]][train_df[["doc","class"]]["class"]==1])
no_class2=unique_doc(train_df[["doc","class"]][train_df[["doc","class"]]["class"]==2])

print("Total no. of observations in class 1 and class 2 are respectively %s and %s" %(no_class1, no_class2))


# In[109]:


#Count of unique documents:
no_doc=unique_doc(train_df[["doc"]])
print("No. of unique documents/datapoints in training set is %s" %no_doc)


# In[14]:


#class probabilities:
prob1=no_class1/no_doc
prob2=no_class2/no_doc


# In[38]:


#Count of words divided by total no in each class-with laplace
for i in range(len(class_1[1])):
    class_1[1][i]=(class_1[1][i]+1)/(no_class1+2)
for j in range(len(class_2[1])):
    class_2[1][j]=(class_2[1][j]+1)/(no_class2+2)      


# In[39]:


#creating dictionary from lists: (word: probability) pair for each class
class1_dict = dict(zip(class_1[0], class_1[1])) 
class2_dict = dict(zip(class_2[0], class_2[1])) 


# In[ ]:


#################################################################
#Most Discriminative Words
#################################################################


# In[110]:


difference=[]
for i in class1_dict.keys():
    difference.append(abs(np.log10(class1_dict[i])-np.log10(class2_dict[i])))


# In[111]:


difference_dict = dict(zip(class_1[0], difference))   
sorted_list=list(difference_dict.values())
sorted_list.sort(reverse=True)


# In[117]:


discrim_words_ID=[]
for i in sorted_list[0:10]:
    discrim_words_ID.append([key  for (key, value) in difference_dict.items() if value == i])
discrim_words_ID.pop(8)
discrim_words_ID


# In[123]:


discrim_words=[]
for i in range(len(discrim_words_ID)):
    for j in range(len(discrim_words_ID[i])):
        discrim_words.append(words[discrim_words_ID[i][j]-1])


# In[125]:


#10 most discriminative words are:
discrim_words




#################################################################
#Training and Testing Accuracy
#################################################################


# In[141]:


train_prediction=nb_classifier(train_df,nwords,class1_dict,class2_dict,prob1,prob2)


# In[142]:


test_prediction=nb_classifier(test_df,nwords,class1_dict,class2_dict,prob1,prob2)


# In[145]:


#Training set accuracy percentage
accuracy(train_df,ytrain,train_prediction)*100


# In[146]:


#Test set accuracy percentage
accuracy(test_df,ytest,test_prediction)*100






