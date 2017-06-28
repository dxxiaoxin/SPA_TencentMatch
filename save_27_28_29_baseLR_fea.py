# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:29:33 2017

@author: Administrator
"""

import pickle 
import pandas as pd


file=open('train_test_concat_and_merge.pkl','rb')
df=pickle.load(file)

dfTrain=df[(df.clickday==27)|(df.clickday==28)|(df.clickday==29)]

dfTest=df[df.clickday==31] 
           
feats = ['creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID',
         'appPlatform','age','gender','residence','connectionType','appCategory',
         'positionID']
crossfeat=[['gender','appID'],['age','appID'],['marriageStatus','appID'],['positionID','connectionType'],
           ['appCategory','appID'],['advertiserID','appID'],['positionID','appID'],
           ['age','gender'],['gender','haveBaby'],['education','appID']]
for i in crossfeat:
    temp=i[0]+i[1]
    dfTrain[temp]=(dfTrain[i[0]].map(lambda x:str(x))+dfTrain[i[1]].map(lambda x:str(x)))
    dfTrain[temp]=dfTrain[temp].map(lambda x:int(x))
    dfTest[temp]=(dfTest[i[0]].map(lambda x:str(x))+dfTest[i[1]].map(lambda x:str(x)))
    dfTest[temp]=dfTest[temp].map(lambda x:int(x)) 
    feats.append(temp)

output1 = open('dfTrain_baseFea.pkl', 'wb')
pickle.dump(dfTrain, output1)
output2=open('dfTest_baseFea.pkl', 'wb')
pickle.dump(dfTest,output2)

