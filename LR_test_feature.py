# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 21:37:23 2017

@author: Administrator
"""
import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
def merge_history_app_action(df,df1,df2):
    df=pd.merge(df,df1,how='left',on='appID')
    df=pd.merge(df,df2,how='left',on='userID')
    del df['appID']
    del df['userID']
    return df
    
def final_merge():
    file1=open('trainFeature.pkl','rb')
    df1=pickle.load(file1)
    file2=open('testFeature.pkl','rb')
    df2=pickle.load(file2)
    #user_action and user_installed_app count
    

    file4=open('appID_installedapps_Counts.pkl','rb')
    df4=pickle.load(file4)
    
    file6=open('user_installedapps_Counts.pkl','rb')
    df6=pickle.load(file6)    
    final_train=merge_history_app_action(df1,df4,df6)
    final_test=merge_history_app_action(df2,df4,df6)
    return final_train,final_test

def split_normalization_discretization(val):
    if 0<=val<0.05:
        return 1
    elif 0.05<=val<0.1:
        return 2
    elif 0.1<=val<0.15:
        return 3
    elif 0.15<=val<0.2:
        return 4
    elif 0.2<=val<0.25:
        return 5
    elif 0.25<=val<0.3:
        return 6
    elif 0.3<=val<0.35:
        return 7
    elif 0.35<=val<0.4:
        return 8
    elif 0.4<=val<0.45:
        return 9
    elif 0.45<=val<0.5:
        return 10
    elif 0.5<=val<0.55:
        return 11
    elif 0.55<=val<0.6:
        return 12
    elif 0.6<=val<0.65:
        return 13
    elif 0.65<=val<0.7:
        return 14
    elif 0.7<=val<0.75:
        return 15
    elif 0.75<=val<0.8:
        return 16
    elif 0.8<=val<0.85:
        return 17
    elif 0.85<=val<0.9:
        return 18
    elif 0.85<=val<0.95:
        return 19
    else:
        return 20
def final_train_test_and_merge():   
    file=open('train_test_concat_and_merge.pkl','rb')
    df=pickle.load(file)
    final_train,final_test=final_merge()     
    #加的一列output1 = open('dfTrain_trickFea.pkl', 'wb')
    
    file1=open('dfTrain_trickFea_test.pkl','rb')
    train_other_three=pickle.load(file1)
    train_other_three=train_other_three.reset_index(drop=True)
    file2=open('dfTest_trickFea_test.pkl','rb')
    test_other_three=pickle.load(file2)
    test_other_three=test_other_three.reset_index(drop=True)
    final_train=pd.concat([final_train,train_other_three],axis=1)
    final_test=pd.concat([final_test,test_other_three],axis=1)    
    
    file3=open('add_trainFeature.pkl','rb')
    train_add_feature=pickle.load(file3)
    train_add_feature=train_add_feature.reset_index(drop=True)
    file4=open('add_testFeature.pkl','rb')
    test_add_feature=pickle.load(file4)
    test_add_feature=test_add_feature.reset_index(drop=True)
    
    final_train=pd.concat([final_train,train_add_feature],axis=1)
    final_test=pd.concat([final_test,test_add_feature],axis=1)
   
    

    
    df_train=df[(df.clickday>=27)&(df.clickday<30)]
    df_test=df[df.clickday==31]
    final_train=final_train.fillna(0.)
    final_test=final_test.fillna(0.)
    return final_train,final_test,df_train,df_test
    
    
    
    



if __name__=="__main__":
    final_train,final_test,df_train,df_test=final_train_test_and_merge()
    cretization_lists=['positionID_connectionType_history_Conversion_Rate',
       'appID_positionID_history_Conversion_Rate',
       'creativeID_positionID_history_Conversion_Rate',
       'gender_positionID_history_Conversion_Rate',
       'advertiserID_positionID_history_Conversion_Rate',
       'residence_positionID_history_Conversion_Rate',
       'residence_creativeID_history_Conversion_Rate',
       'residence_marriageStatus_history_Conversion_Rate',
       'age_positionID_history_Conversion_Rate',
       'positionID_marriageStatus_history_Conversion_Rate',
       'hometown_positionID_history_Conversion_Rate',
       'age_creativeID_history_Conversion_Rate',
       'telecomsOperator_positionID_history_Conversion_Rate',
       'education_positionID_history_Conversion_Rate',
       'camgaignID_connectionType_history_Conversion_Rate',
       'creativeID_connectionType_history_Conversion_Rate',
       'adID_connectionType_history_Conversion_Rate',
       'positionID_clickhour_history_Conversion_Rate',
       'positionID_appCate_Max_history_Conversion_Rate',
       'positionID_appCategory_history_Conversion_Rate',
       'age_gender_education_marriageStatus_positionID_history_Conversion_Rate',
       'age_gender_education_marriageStatus_connectionType_history_Conversion_Rate',
       'age_gender_education_marriageStatus_appID_history_Conversion_Rate',
       'age_gender_education_marriageStatus_creativeID_history_Conversion_Rate',
       'age_gender_education_marriageStatus_appPlatform_history_Conversion_Rate',
       'age_gender_education_marriageStatus_sitesetID_history_Conversion_Rate',
       'age_gender_education_marriageStatus_clickhour_history_Conversion_Rate',
       'age_gender_education_marriageStatus_appCate_Max_history_Conversion_Rate',
       'age_gender_education_marriageStatus_appCategory_history_Conversion_Rate',
       'appID_connectionType_history_Conversion_Rate',
       'appID_residence_history_Conversion_Rate', 'positionID_Click_Count',
       'positionID_history_Conversion_Rate', 'userID_Click_Count',
       'userID_history_Conversion_Rate',
       'appID_installed_Count', 'user_installed_app_Count', 'firstTrick',
       'secondTrick', 'thirdTrick', 'minTimeDiff', 'maxTimeDiff',
       'appID_advertiserID_appCate_Max_appPlatform_history_Conversion_Rate',
       'advertiserID_appCate_Max_appPlatform_positionID_history_Conversion_Rate',
       'advertiserID_appCate_Max_appPlatform_connectionType_history_Conversion_Rate',
       'advertiserID_appID_diff_convert_click',
       'age_gender_education_marriageStatus_appID_diff_convert_click',
       'connectionType_appID_diff_convert_click',
       'positionID_connectionType_diff_convert_click']
    

    final_train.ix[:,cretization_lists]=final_train.ix[:,cretization_lists].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    final_train.ix[:,cretization_lists]=final_train.ix[:,cretization_lists].applymap(split_normalization_discretization)
    
    final_test.ix[:,cretization_lists]=final_test.ix[:,cretization_lists].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    final_test.ix[:,cretization_lists]=final_test.ix[:,cretization_lists].applymap(split_normalization_discretization)   
    enc = OneHotEncoder()
    k=0    
    for i,feat in enumerate(final_train.columns):
        k+=1
        x_=enc.fit_transform(np.vstack((final_train[feat].values.reshape(-1,1),final_test[feat].values.reshape(-1,1))))
        x_train = enc.transform(final_train[feat].values.reshape(-1, 1))
        x_test = enc.transform(final_test[feat].values.reshape(-1, 1))
        if i == 0:
            X_train, X_test = x_train, x_test
        
        else:
            X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
    # model training
    lr = LogisticRegression()
    lr.fit(X_train, df_train.label.values)
    proba_test = lr.predict_proba(X_test)[:,1]

    # submission
    df = pd.DataFrame({"instanceID": df_test["instanceID"].values.astype(int), "proba": proba_test})
    df.sort_values("instanceID", inplace=True)

    df.to_csv("submission.csv", index=False)
    