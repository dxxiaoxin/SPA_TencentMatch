

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:07:15 2017

@author: Administrator
"""



import zipfile
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import scipy as sp
import xgboost as xgb
import pickle


#base ID feature
def baseIDFeature(df):
    base_lists=['clickday']
    
    df=df[(df.clickday>=27)]
    '''
    crossfeat=[['positionID','connectionType']]
    for i in crossfeat:
        temp=i[0]+i[1]
        df[temp]=(df[i[0]].map(lambda x:str(x))+df[i[1]].map(lambda x:str(x)))
        df[temp]=df[temp].map(lambda x:int(x))
        base_lists.append(temp)
    '''
    df1=df.ix[:,base_lists]

    #df1.creativeID=df1.creativeID.astype('category')
    #df1.userID=df1.userID.astype('category')
    
    #df1.adID=df1.adID.astype('category')
    #df1.camgaignID=df1.camgaignID.astype('category')
    #df1.appID=df1.appID.astype('category')
    return df1

# 前七天的历史记录作为后七天训练集的样本，统计点击数，转化数，转化率
def CombinTestTwoFeas(df,fea1,fea):
    df_hist=df[(df.clickday>=20)&(df.clickday<27)]
    df_train=df[(df.clickday>=27)]
    #点击数
    '''
    df_hist['%s_%s__Click_Count'%(fea1,fea)]=1
    df1=df_hist.ix[:,['%s'%fea1,'%s'%fea,'%s_%s__Click_Count'%
                      (fea1,fea)]].groupby(['%s'%fea1,'%s'%fea]).agg('sum').reset_index()
    df_hist['%s_%s_Convision_Count'%(fea1,fea)]=1
    df2=df_hist[df_hist.label==1][['%s'%fea1,'%s'%fea,'%s_%s_Convision_Count'%
                        (fea1,fea)]].groupby(['%s'%fea1,'%s'%fea]).agg('sum').reset_index()
    
    '''
    df_hist['%s_%s_history_Conversion_Rate'%(fea1,fea)]=1
    t1=df_hist[['%s'%fea1,'%s'%fea,'%s_%s_history_Conversion_Rate'%
                (fea1,fea)]].groupby(['%s'%fea1,'%s'%fea]).agg('sum')
    t2=df_hist[df_hist.label==1][['%s'%fea1,'%s'%fea,'%s_%s_history_Conversion_Rate'%
                (fea1,fea)]].groupby(['%s'%fea1,'%s'%fea]).agg('sum')
    df3=(t2/t1)*1000
    df3=df3.reset_index()# 把索引变为列
    
    df_train=df_train.ix[:,['%s'%fea1,'%s'%fea]]
    #df_train=pd.merge(df_train,df1,how='left',on=['%s'%fea1,'%s'%fea])
    #df_train=pd.merge(df_train,df2,how='left',on=['%s'%fea1,'%s'%fea])
    df_train=pd.merge(df_train,df3,how='left',on=['%s'%fea1,'%s'%fea])
    
    df_train['%s_%s_history_Conversion_Rate'%(fea1,fea)]=df_train['%s_%s_history_Conversion_Rate'%(fea1,fea)].fillna(0.47613/(0.47613+26.996)*1000)
    
    df_train=df_train.drop(['%s'%fea1,'%s'%fea],axis=1)
    return df_train
def CombinTestOneFeas(df,fea):
    df_hist=df[(df.clickday>=20)&(df.clickday<27)]
    df_train=df[(df.clickday>=27)]
    #点击数
    
    df_hist['%s_Click_Count'%(fea)]=1
    df1=df_hist.ix[:,['%s'%fea,'%s_Click_Count'%
                      (fea)]].groupby(['%s'%fea]).agg('sum').reset_index()
    '''
    df_hist['%s_Convision_Count'%(fea)]=1
    df2=df_hist[df_hist.label==1][['%s'%fea,'%s_Convision_Count'%
                        (fea)]].groupby(['%s'%fea]).agg('sum').reset_index()
    
    '''
    df_hist['%s_history_Conversion_Rate'%(fea)]=1
    t1=df_hist[['%s'%fea,'%s_history_Conversion_Rate'%
                (fea)]].groupby(['%s'%fea]).agg('sum')
    t2=df_hist[df_hist.label==1][['%s'%fea,'%s_history_Conversion_Rate'%
                (fea)]].groupby(['%s'%fea]).agg('sum')
    df3=(t2/t1)*1000
    df3=df3.reset_index()# 把索引变为列
    
    df_train=df_train.ix[:,['%s'%fea]]
    df_train=pd.merge(df_train,df1,how='left',on=['%s'%fea])
    #df_train=pd.merge(df_train,df2,how='left',on=['%s'%fea])
    df_train=pd.merge(df_train,df3,how='left',on=['%s'%fea])
    #df_train['%s_history_Conversion_Rate'%(fea)]=df_train['%s_history_Conversion_Rate'%(fea)].fillna(
             #np.mean(df_train['%s_history_Conversion_Rate'%(fea)]))
    df_train=df_train.drop(['%s'%fea],axis=1)
    df_train['%s_history_Conversion_Rate'%(fea)]=df_train['%s_history_Conversion_Rate'%(fea)].fillna(0.47613/(0.47613+26.996)*1000)

    df_train['%s_Click_Count'%(fea)]=df_train['%s_Click_Count'%(fea)].map(lambda x:np.log(x**2))

    return df_train
    
def time_diff_fea(df,fea1,fea):

    df_hist=df[(df.clickday>=20)&(df.clickday<27)]
    df_train=df[(df.clickday>=27)]
    #历史的转化与最后一次点击的差值的均值
    df_hist['%s_%s_diff_convert_click'%(fea1,fea)]=df_hist.conversionTime-df_hist.clickTime
    df1=df_hist[df_hist.label==1][['%s'%fea1,'%s'%fea,'%s_%s_diff_convert_click'%(fea1,fea)]]
    df2=df1.groupby(['%s'%fea1,'%s'%fea]).agg('mean').reset_index()
    df_train=df_train.ix[:,['%s'%fea1,'%s'%fea]]
    df_train=pd.merge(df_train,df2,how='left',on=['%s'%fea1,'%s'%fea])
    df_train=df_train.drop(['%s'%fea1,'%s'%fea],axis=1)
    return df_train
    



def Merge_Combine_Feature(df):
    combine_Feature_lists=[['appID','advertiserID_appCate_Max_appPlatform'],['advertiserID_appCate_Max_appPlatform','positionID'],
                           
                           ['advertiserID_appCate_Max_appPlatform','connectionType'],['advertiserID_appCate_Max_appPlatform','positionType'],
                           ['advertiserID_appCate_Max_appPlatform','age_gender_education_marriageStatus'],['advertiserID_appCate_Max_appPlatform','sitesetID']]
    '''
    combine_Feature_lists=[['positionID','connectionType'],['age_gender_education','positionID'],
                           ['residence','positionID'],['age','hometown'],['camgaignID','connectionType'],
                           ['age_gender_education','appPlatform'],['age','residence'],['age_gender_education','camgaignID'],
                           ['age_gender_education','connectionType'],['appCategory','positionID'],
                           ['adID','connectionType'],['age_gender_education','positionType'],['creativeID','connectionType'],
                           ['appID','positionID'],['creativeID','positionID'],['creativeID','positionID'],
                           ['age_gender_education','sitesetID'],['advertiserID','positionID'],['age_gender_education','appCate_Max'],
                           ['age_gender_education','clickhour'],['positionID','clickhour'],['positionID','appCate_Max'],
                           ['age_gender_education','appID'],['age_gender_education','creativeID'],['age_gender_education','appPlatform']]
    
    '''
    df_fea=CombinTestTwoFeas(df,'appID','advertiserID_appCate_Max_appPlatform')
    for fealist in combine_Feature_lists[1:]:
        df_fea=pd.concat([df_fea,CombinTestTwoFeas(df,fealist[0],fealist[1])],axis=1)
       
        
    #df_fea=pd.concat([df_fea,CombinTestOneFeas(df,'creativeID')],axis=1)
    return df_fea

def Merge_diff_fea(df):
    combine_Feature_lists=[['advertiserID','appID'],['age_gender_education_marriageStatus','appID'],
                           ['connectionType','appID'],['positionID','connectionType'],['age_gender_education_marriageStatus','positionID'],
                           ['positionID','appID'],['positionID','residence'],['positionID','appCate_Max'],['positionID','advertiserID']]
    df_fea=time_diff_fea(df,'advertiserID','appID')
    for fealist in combine_Feature_lists[1:]:
        df_fea=pd.concat([df_fea,time_diff_fea(df,fealist[0],fealist[1])],axis=1)
    return df_fea

def split_train_test(df_combinFea,df_base,df_diff):
    df_base_combinFea=pd.concat([df_combinFea,df_base,df_diff],axis=1)
    df_base_combinFea_train=df_base_combinFea[df_base_combinFea.clickday<30]
    df_base_combinFea_test=df_base_combinFea[df_base_combinFea.clickday==31]
    del df_base_combinFea_train['clickday']
    del df_base_combinFea_test['clickday']
    
    output1 = open('add_trainFeature.pkl', 'wb')
    pickle.dump(df_base_combinFea_train, output1)
    output2 = open('add_testFeature.pkl', 'wb')
    pickle.dump(df_base_combinFea_test, output2)
    return df_base_combinFea_train,df_base_combinFea_test



if __name__=="__main__":
    
    file=open('train_test_concat_and_merge.pkl','rb')
    df=pickle.load(file)
    df_diff=Merge_diff_fea(df)    
    df_diff=df_diff.reset_index(drop=True)
    df_combinFea=Merge_Combine_Feature(df)
    df_combinFea=df_combinFea.reset_index(drop=True)
    df_base=baseIDFeature(df)
    df_base=df_base.reset_index(drop=True)
    train,test=split_train_test(df_combinFea,df_base,df_diff)
    
    
    
    