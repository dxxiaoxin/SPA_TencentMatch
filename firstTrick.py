# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:25:19 2017

@author: Administrator
"""

import pandas as pd
import pickle 
import numpy as np


def firstTrick(train_test):
    df=train_test.copy()
    df['click_day_hour_minute']=df.clickTime.map(lambda x:int(str(x)[:6]))
    df=df.ix[:,['click_day_hour_minute','userID','appID','positionID','creativeID']]
    df1=df.drop_duplicates(keep=False)
    df_rep=df1.index                     #非重复记录的行
    df['firstTrick']=0
    df.ix[df.index.isin(df_rep),'firstTrick']=1        #非重复
    df2=df.drop_duplicates(keep='first')            #除第一个
    df_rep_last=df2.index
    df_last=df[~df.index.isin(df_rep_last)].index        #第二个重复，到最后一个
    df.ix[df.index.isin(df_last),'firstTrick']=2       #重复非第一个 
    
    df=df.ix[:,['firstTrick']]
    return df
    
def secondTrick(train_test):
    df=train_test.copy()
    #非重复=1，重复非最后一个=0    重复最后一个=2    重复第一个=3
    df['click_day_hour_minute']=df.clickTime.map(lambda x:int(str(x)[:6]))
    df=df.ix[:,['click_day_hour_minute','userID','appID','positionID','creativeID']]
    df1=df.drop_duplicates(keep=False)
    df_rep=df1.index                     #非重复记录的行
    df['secondTrick']=0
    df.ix[df.index.isin(df_rep),'secondTrick']=1
    df2=df.drop_duplicates(keep='last')
    df_rep_last=df2.index
    df_last=df[~df.index.isin(df_rep_last)].index           
    df.ix[df.index.isin(df_last),'secondTrick']=2        
    df3=df.drop_duplicates(keep='first')
    df_rep_first=df3.index
    df_first=df[~df.index.isin(df_rep_first)].index
    df.ix[df.index.isin(df_first),'secondTrick']=3
    df=df.ix[:,['secondTrick']]
    return df

    
def thirdTrick(train_test):
    df=train_test.copy()    
    
    df=df.ix[:,['userID','appID','positionID','creativeID','camgaignID']]
    df1=df.drop_duplicates(keep=False)
    df_rep=df1.index                     #非重复记录的行
    df['thirdTrick']=0
    df.ix[df.index.isin(df_rep),'thirdTrick']=1
    df2=df.drop_duplicates(keep='last')
    df_rep_last=df2.index
    df_last=df[~df.index.isin(df_rep_last)].index           
    df.ix[df.index.isin(df_last),'thirdTrick']=2        
    df3=df.drop_duplicates(keep='first')
    df_rep_first=df3.index
    df_first=df[~df.index.isin(df_rep_first)].index
    df.ix[df.index.isin(df_first),'thirdTrick']=3    
    df=df.ix[:,['thirdTrick']]
    return df

    
def forthTrick(train_test):
    df=train_test.copy()
    df=df.ix[:,['userID','appID','clickTime','positionID','creativeID']]
    df1=df.drop_duplicates(['userID','appID','positionID','creativeID'],keep=False)
    df_rep=df1.index
    df2=df[~df.index.isin(df_rep)]             #重复记录的行

    df3=df2.groupby(['userID','appID','positionID','creativeID'])['clickTime'].apply(lambda x:x-min(x))
    df3=df3.rename(columns={'minTimeDiff':'clickTime'})

    df4=df2.groupby(['userID','appID','positionID','creativeID'])['clickTime'].apply(lambda x:max(x)-x)

    df4=df4.rename(columns={'maxTimeDiff':'clickTime'})    
    return df3,df4
    
if __name__=="__main__":
    file=open('train_test_concat_and_merge.pkl','rb')
    train_test=pickle.load(file)
    train_test=train_test[train_test.clickday>=27]
    df1=firstTrick(train_test)
    
    df2=secondTrick(train_test)
    df3=thirdTrick(train_test)
    df4,df5=forthTrick(train_test)
    
    train_test=pd.concat([train_test,df1,df2,df3,df4,df5],axis=1)
    train=train_test[(train_test.clickday>=27)&(train_test.clickday<30)]  #27,28,29
    test=train_test[train_test.clickday==31]
    train=train.ix[:,['firstTrick','secondTrick','thirdTrick','minTimeDiff','maxTimeDiff']]
    test=test.ix[:,['firstTrick','secondTrick','thirdTrick','minTimeDiff','maxTimeDiff']]
    
    output1 = open('dfTrain_trickFea.pkl', 'wb')
    pickle.dump(train, output1)
    output2=open('dfTest_trickFea.pkl', 'wb')
    pickle.dump(test,output2)

    '''
#非重复=1，重复非最后一个=0    重复最后一个=2
df=test.copy()
df['click_day_hour_minute']=df.clickTime.map(lambda x:int(str(x)[:6]))
df=df.ix[:,['click_day_hour_minute','userID','appID','positionID','creativeID']]
df1=df.drop_duplicates(keep=False)
df_rep=df1.index                     #非重复记录的行
df['firstTrick']=0
df.ix[df.index.isin(df_rep),'firstTrick']=1
df2=df.drop_duplicates(keep='last')
df_rep_last=df2.index
df_last=df[~df.index.isin(df_rep_last)].index
df.ix[df.index.isin(df_last),'firstTrick']=2

'''

'''
#非重复=1，重复非最后一个=0    重复最后一个=2    重复第一个=3
df=test.copy()
df['click_day_hour_minute']=df.clickTime.map(lambda x:int(str(x)[:6]))
df=df.ix[:,['click_day_hour_minute','userID','appID','positionID','creativeID']]
df1=df.drop_duplicates(keep=False)
df_rep=df1.index                     #非重复记录的行
df['secondTrick']=0
df.ix[df.index.isin(df_rep),'secondTrick']=1
df2=df.drop_duplicates(keep='last')
df_rep_last=df2.index
df_last=df[~df.index.isin(df_rep_last)].index           
df.ix[df.index.isin(df_last),'secondTrick']=2        
df3=df.drop_duplicates(keep='first')
df_rep_first=df3.index
df_first=df[~df.index.isin(df_rep_first)].index
df.ix[df.index.isin(df_first),'secondTrick']=3

'''

'''
#没有加clickTime的重复数据
df=test.copy()
df=df.ix[:,['userID','appID','positionID','creativeID','camgaignID']]
df1=df.drop_duplicates(keep=False)
df_rep=df1.index                     #非重复记录的行
df['thirdTrick']=0
df.ix[df.index.isin(df_rep),'thirdTrick']=1
df2=df.drop_duplicates(keep='last')
df_rep_last=df2.index
df_last=df[~df.index.isin(df_rep_last)].index           
df.ix[df.index.isin(df_last),'thirdTrick']=2        
df3=df.drop_duplicates(keep='first')
df_rep_first=df3.index
df_first=df[~df.index.isin(df_rep_first)].index
df.ix[df.index.isin(df_first),'thirdTrick']=3
'''


'''
df=test.copy()
df=df.ix[:,['userID','appID','clickTime','positionID','creativeID']]
df1=df.drop_duplicates(['userID','appID','positionID','creativeID'],keep=False)
df_rep=df1.index
df2=df[~df.index.isin(df_rep)]             #重复记录的行

df3=df2.groupby(['userID','appID','positionID','creativeID'])['clickTime'].apply(lambda x:x-min(x))
df3=df3.rename(columns={'minTimeDiff':'clickTime'})

df4=df2.groupby(['userID','appID','positionID','creativeID'])['clickTime'].apply(lambda x:max(x)-x)

df4=df4.rename(columns={'maxTimeDiff':'clickTime'})

'''

