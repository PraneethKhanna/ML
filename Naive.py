# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import confusion_matrix
# First we read in the dataset
Dataset = pd.read_csv("C:/Users/pxb171530/Downloads/DEMO/amazon_cells_labelled.txt", sep="\t", header = None, names=['Tweet', 'Sentiment'] )

# pre-processinng
Dataset["Tweet"]=Dataset["Tweet"].apply(lambda x: x.lower())
Dataset["Tweet"]=Dataset["Tweet"].apply(lambda x: re.sub("'ve",' have',x))
Dataset["Tweet"]=Dataset["Tweet"].apply(lambda x: re.sub("didn't",'did not',x))
Dataset["Tweet"]=Dataset["Tweet"].apply(lambda x: re.sub("you're",'you are',x))
Dataset["Tweet"]=Dataset["Tweet"].apply(lambda x: re.sub("i'm",'i am',x))
Dataset["Tweet"]=Dataset["Tweet"].apply(lambda x: re.sub("it's",'it is',x))
Dataset["Tweet"]=Dataset["Tweet"].apply(lambda x: re.sub(r"['#*<>();$%&\",""-_+!?@:.]","",x))


Dataset["Tweet"]=Dataset["Tweet"].apply(lambda x: re.sub("a",'',x))


x_train,x_test,y_train,y_test= train_test_split(Dataset.iloc[:,:2] ,Dataset['Sentiment'], test_size=0.2, random_state=0)

Dataset = x_train


dictionary_freq = {}
for line in Dataset["Tweet"]:
    for word in line.split():
        if word not in dictionary_freq:
            dictionary_freq[word] = 1
        else:
            dictionary_freq[word] += 1


word_id = 1            
index_dictionary = {}
for words,count in dictionary_freq.items():
    if(count < 200 and count > 1):
        index_dictionary[words] = word_id
        word_id +=1
 

indexed=[]            
for line in Dataset["Tweet"]:
    sen=[]
    for word in line.split():
        if word in index_dictionary:
            sen.append(index_dictionary[word])
    indexed.append(sen)
            
Dataset["indexed"] = indexed

'''
Naive Bayes
Conditional Probability
P(A|B1,B2,B3) = P(B1,B2,B3|A)*P(A)/P(B1,B2,B3)


P(B1,B2,B3|A) = P(B1|A)*P(B2|A)*P(B3|A)
'''


neg = Dataset.loc[Dataset['Sentiment'] == 0]
neg = neg.reset_index()
pos = Dataset.loc[Dataset['Sentiment'] == 1]
pos = pos.reset_index()

p = len(pos)
q = len(neg)

prob_pos = p/(p+q)
prob_neg = q/(p+q)


dictionary_pos_freq = {}
for row in pos["indexed"]:
    for index in row:
       if index not in dictionary_pos_freq:
            dictionary_pos_freq[index] = 1
       else:
            dictionary_pos_freq[index] += 1    
        
       
dictionary_neg_freq = {}
for row in neg["indexed"]:
    for index in row:
       if index not in dictionary_neg_freq:
            dictionary_neg_freq[index] = 1
       else:
            dictionary_neg_freq[index] += 1              






def target(sen):
    idx_sen = []
    for word in sen.split():
        if word in index_dictionary:
            idx_sen.append(index_dictionary[word])
    cond_prob_pos =  1      
    for i in range(0,len(idx_sen)):
        if idx_sen[i] in dictionary_pos_freq:
            cond_prob_pos = cond_prob_pos*(dictionary_pos_freq[idx_sen[i]]/len(pos))
        else:
            cond_prob_pos = cond_prob_pos*(1/len(pos))
     
       
    cond_prob_neg =  1      
    for i in range(0,len(idx_sen)):
        if idx_sen[i] in dictionary_neg_freq:
            cond_prob_neg = cond_prob_neg*(dictionary_neg_freq[idx_sen[i]]/len(neg))
        else:
            cond_prob_neg = cond_prob_neg*(1/len(neg))
    
    if(cond_prob_pos > cond_prob_neg):
        result = 1
    else:
        result = 0
    return result

'''
predictions = []
for row in Dataset['Tweet']:
    predictions.append(target(row))

Dataset['predictions'] = predictions


match = 0
for i in range(0,len(Dataset)):
    if (Dataset['predictions'][i] == Dataset['Sentiment'][i]):
        match += 1    
accuracy = match / len(Dataset)
print(accuracy)
'''



predictions = []
for row in x_test['Tweet']:
    predictions.append(target(row))

x_test['predictions'] = predictions



x_test = x_test.reset_index()
match = 0
for i in range(0,len(x_test)):
    
    if (x_test['predictions'][i] == x_test['Sentiment'][i]):
        match += 1 
       
accuracy = match / len(x_test)
print("accuracy is ",accuracy)

import seaborn as sn
import matplotlib.pyplot as plt

CF_dataFrame = pd.DataFrame(confusion_matrix(y_test, predictions,labels = [0,1]))
print("confusion matrix ",confusion_matrix(y_test, predictions,labels = [0,1]))
plt.figure(figsize = (2,2))
sn.heatmap(CF_dataFrame, annot=True)
plt.xlabel("predictions")
plt.ylabel("actual")

print("precision",CF_dataFrame[1][1]/(CF_dataFrame[1][1]+CF_dataFrame[0][1]))

print("recall",CF_dataFrame[1][1]/(CF_dataFrame[1][1]+CF_dataFrame[1][0]))

















