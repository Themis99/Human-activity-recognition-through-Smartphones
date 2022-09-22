# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""
#prin treksei katharizoume tin konsola
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter 

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA



from pathlib import Path

def replace_sensor_errors(data):
    #fix values greater than 1000
    
    for i in range(data.shape[0]):
        
        if(i == 0 and data[i]>1000):
            if(data[i+1]<1000):
                data[i] = data[i+1]
            else:
                k=i+1
                while(data[k]>1000):
                    k = k+1
                    assert k==(data.shape[0]-1) , 'All values in this dataset are errors'
                data[i] = data[k]
            
        if(data[i] > 1000 and i>0):
            data[i] = data[i-1]
                               
    return data

def most_frequent(List): 
    
    occurence_count = Counter(List)
    
    return occurence_count.most_common(1)[0][0]
    
def preprocess_dataset(dataset):
    #preprocesses each dataset : 
    #1) choose right pocket 
    #2) keep only the magnitude of the vector 
    #3) replace error of the sensor(values greater than 1000)
    
    #keep only values for right pocket
    dataset.index = range(0,dataset.shape[0])
    
    
    right_pocket_values = np.array(dataset[['Ax.1','Ay.1','Az.1']],dtype='float')
    #compute tha magnitude (from list to numpy array)
    magnitude = np.array([np.linalg.norm(right_pocket_values[w,:]) for w in range(0,right_pocket_values.shape[0])],dtype='float')

    #replace sensor errors
    magnitude = replace_sensor_errors(magnitude)

    #create the final dataset(after the preprocessing) with the target
    participant = pd.DataFrame()
    participant['a'] = magnitude
    participant['y'] = dataset['Unnamed: 69']
    

    
    mask = participant['y'] == 'upsatirs'
    participant['y'].loc[mask] = 'upstairs'
    
        
    #add 1000 samples to the end of dataset , because of the window
    new_participant = participant.reindex(index=participant.index[::-1])
    participant = pd.concat([participant,new_participant.iloc[0:1000,:]])
    #fix the indexing in dataframe
    participant.index = range(participant.shape[0])
    #print(participant.iloc[62995:63005,:])

    return participant


def featureExtraction(participant):
    
    
    list_mean=[]
    list_std=[]
    list_skew=[]
    list_max=[]
    list_min=[]
    list_difference=[]
    list_y=[]
    
    
    df = pd.DataFrame()

    
    for w in range(0,participant.shape[0]-1000,50):
        end = w+1000    

        list_mean.append(np.mean(participant.iloc[w:end,0]))
        list_std.append(np.std(participant.iloc[w:end,0]))
        list_skew.append(skew(participant.iloc[w:end,0]))
        list_max.append(np.max(participant.iloc[w:end,0]))
        list_min.append(np.min(participant.iloc[w:end,0]))
        list_difference.append(np.max(participant.iloc[w:end,0]) - np.min(participant.iloc[w:end,0]))
        
        #f,p = signal.welch(participant.iloc[w:end,0],window='hamming', nperseg=N, noverlap=(N-M)) #,window='hamming', nperseg=N, noverlap=(N-M)
        f,p = signal.welch(participant.iloc[w:end,0],nperseg=128)
        
        
        if (w==0):
            n=f.shape[0]
            welch_lists =[[] for i in range(n)]

        
        for t in range(0,n):
            welch_lists[t].append(p[t]) 
             
        
        list_y.append(most_frequent(participant['y'][w:end]))

    
    df['Mean'] = list_mean 
    df['Std'] = list_std
    df['Skew'] = list_skew
    df['Max'] = list_max
    df['Min'] = list_min
    df['Difference']=list_difference
    

    for k in range(0,n):
        df['Welch'+str(k)] = welch_lists[k]
    
    df['y'] = list_y
    
    return df



def LOSO_SVM():
    
    cm = []
    acc_list=[]
    
    yTest = []
    yPred = []


    for i in range(0,10):
         
        current_Participant = pd.read_csv(filename[i],header=[1])
        train = pd.DataFrame()
        
        
        for t in range(0,10):
            if (t==i): 
                continue
            
            
            train = pd.concat([train,pd.read_csv(filename[t],header=[1])])        
        
        
        participant = preprocess_dataset(current_Participant)
        
        
        train = preprocess_dataset(train)
        
        
        test = featureExtraction(participant)
        train_participant_features = featureExtraction(train)
        
                        
        X_train = train_participant_features.iloc[:,0:-1]
        X_test = test.iloc[:,0:-1]
        
        y_train = train_participant_features.iloc[:,-1]
        y_test = test.iloc[:,-1]
        

        # MinMaxScaler gia Normalized to [0,1]
        scaler = MinMaxScaler().fit(X_train)
        #scaler = StandardScaler().fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)
        
        
        ################PCA analysis for eah participant#######################
        #you must uncomment the following lines for PCA analysis
        # Make an instance of the Model
        #pca = PCA(.95)
        #pca.fit(X_train_norm)
        #print(f'{pca.n_components_} are the most significant features for Participant{i+1}')

        #print(pca.components_)
        #print(explained_variance_.shape)
        #print(explained_variance_)

        #X_train_norm = pca.transform(X_train_norm)
        #X_test_norm = pca.transform(X_test_norm)
        #######################################################################
        
        
        svc = svm.SVC( kernel='rbf',decision_function_shape='ovo',C=1,gamma=1/X_train_norm.shape[1])
        svc.fit(X_train_norm, y_train)
        y_hat = svc.predict(X_test_norm)
        
        yTest.extend(y_test)
        yPred.extend(y_hat)
        
        
        acc = accuracy_score(y_test, y_hat)
        print(f'SVM accuracy for Participant {i+1} is {acc}')
        print(f'Confusion matrix for Participant {i+1} is :')
        cfm = confusion_matrix(y_test, y_hat, labels= ['walking', 'standing', 'jogging','sitting', 'biking', 'upstairs', 'downstairs'])
        print(cfm)
        
        
        
        
#        
#        list_of_target_names  = train_participant_features['y'].drop_duplicates().tolist()
#        print(list_of_target_names)      
#        
#        plt.figure(figsize=(7, 6))
#        plt.title(f'Confusion matrix for Participant {i+1}', fontsize=16)
#        plt.imshow(cfm)
#        plt.xticks(np.arange(len(list_of_target_names)), labels = list_of_target_names, rotation=45, fontsize=12)
#        plt.yticks(np.arange(len(list_of_target_names)), labels = list_of_target_names, fontsize=12)
#        
#        fmt = 'd'
#        for i in range(cfm.shape[0]):
#            for j in range(cfm.shape[1]):
#                plt.text(j, i, format(cfm[i, j], fmt), ha="center", va="center",color="white",fontsize=18)
#        plt.tight_layout()
#                
#        plt.colorbar()
#        plt.show()
        
        cm.append(cfm)
        acc_list.append(acc)
        
    return cm ,yTest, yPred



if __name__ == '__main__':

    #check if dataset folder exists in current directory
    assert Path('dataset').exists(), 'In order to run this program you must create a folder \dataset in working directory, with all the csv files inside'

    #check for missing csv files
    filename = []
    for i in range(0,10):
        assert Path('dataset/Participant_' + str(i+1) + '.csv').exists(), 'File: ' + 'dataset/Participant_' + str(i+1) + '.csv' + ' is missing'
        #keep all filenames in a list
        filename.append('dataset/Participant_' + str(i+1) + '.csv')

    cm , yTest, yPred = LOSO_SVM()
    
    
    complete_conf_matrix = cm[0]+cm[1]+cm[2]+cm[3]+cm[4]+cm[5]+cm[6]+cm[7]+cm[8]+cm[9]
    
    list_of_target_names = ['walking', 'standing', 'jogging','sitting', 'biking', 'upstairs', 'downstairs']
    plt.figure(figsize=(7, 6))
    plt.title(f'Confusion matrix for all Participants', fontsize=16)
    plt.imshow(complete_conf_matrix)
    plt.xticks(np.arange(len(list_of_target_names)), labels = list_of_target_names, rotation=45, fontsize=12)
    plt.yticks(np.arange(len(list_of_target_names)), labels = list_of_target_names, fontsize=12)
    
    
    fmt = 'd'
    #thresh = cm.max() / 2.
    for i in range(complete_conf_matrix.shape[0]):
        for j in range(complete_conf_matrix.shape[1]):
            plt.text(j, i, format(complete_conf_matrix[i, j], fmt), ha="center", va="center",color="white",fontsize=18)
    plt.tight_layout()
    plt.autoscale()  
    
    plt.colorbar()
    plt.show()    
    middle_acc= complete_conf_matrix.diagonal()/complete_conf_matrix.sum(axis=1)
    print(middle_acc)
    print('Confusion matrix for all Participants is:')
    print(complete_conf_matrix)
    print('Classification statistics are:')
    print(classification_report(yTest, yPred, target_names=list_of_target_names))
    print('Accuracy is :')
    print(accuracy_score(yTest, yPred))
    
    
    
    
    complete_conf_matrix3=complete_conf_matrix.copy()
    complete_conf_matrix4 = np.zeros((6,7),dtype=int)
    ind1=1
    ind2=3
    for i in range(complete_conf_matrix3.shape[0]-1):
        if (i<ind1 or (i>ind1 and i<ind2)):
            complete_conf_matrix4[i,:]=complete_conf_matrix3[i,:]

        elif (i>ind1 and (i+1)>ind2):
            complete_conf_matrix4[i,:]=complete_conf_matrix3[i+1,:]
        
        
        if (i==ind1):
            complete_conf_matrix4[i,:]=np.sum([complete_conf_matrix3[i,:],complete_conf_matrix3[ind2,:]],axis=0)
        
        
    complete_conf_matrix3=complete_conf_matrix4.copy()
    complete_conf_matrix4 = np.zeros((6,6),dtype=int)
    for i in range(complete_conf_matrix3.shape[1]-1):
        
        if(i<ind1 or (i>ind1 and (i)<ind2)):
            complete_conf_matrix4[:,i]=complete_conf_matrix3[:,i]
        elif (i>ind1 and (i+1)>ind2):
            complete_conf_matrix4[:,i]=complete_conf_matrix3[:,i+1]
            
        if (i==ind1):
            complete_conf_matrix4[:,i]=np.sum([complete_conf_matrix3[:,i],complete_conf_matrix3[:,ind2]],axis=0).T
    


    
    list_of_target_names = ['walking', 'standing_sitting', 'jogging', 'biking', 'upstairs', 'downstairs']
    plt.figure(figsize=(7, 6))
    plt.title(f'Confusion matrix for all Participants', fontsize=16)
    plt.imshow(complete_conf_matrix4)
    plt.xticks(np.arange(len(list_of_target_names)), labels = list_of_target_names, rotation=45, fontsize=12)
    plt.yticks(np.arange(len(list_of_target_names)), labels = list_of_target_names, fontsize=12)
    
    fmt = 'd'
    #thresh = cm.max() / 2.
    for i in range(complete_conf_matrix4.shape[0]):
        for j in range(complete_conf_matrix4.shape[1]):
            plt.text(j, i, format(complete_conf_matrix4[i, j], fmt), ha="center", va="center",color="white",fontsize=18)
    plt.tight_layout()
      
    plt.autoscale()
    plt.colorbar()
    plt.show()    
    acc= np.sum(np.diag(complete_conf_matrix4))/np.sum(complete_conf_matrix4)
    print(acc)
    TP = np.diag(complete_conf_matrix4)
    FP = np.sum(complete_conf_matrix4, axis=0) - TP
    FN = np.sum(complete_conf_matrix4, axis=1) - TP
    num_classes = 6
    TN = []
    for i in range(num_classes):
        temp = np.delete(complete_conf_matrix4, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
        
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*precision*recall/(precision+recall)
    print(precision)
    print(recall)
    print(f1_score)
    
    middle_acc2= complete_conf_matrix4.diagonal()/complete_conf_matrix4.sum(axis=1)
    print(middle_acc2)
