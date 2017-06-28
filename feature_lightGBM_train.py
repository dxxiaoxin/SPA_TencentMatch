# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:25:08 2017

@author: Administrator
"""

import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
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



if __name__=="__main__":
    file=open('train_test_concat_and_merge.pkl','rb')
    df=pickle.load(file)
    final_train,final_test=final_merge()
    
    #加的一列output1 = open('dfTrain_trickFea.pkl', 'wb')
    
    file1=open('dfTrain_trickFea.pkl','rb')
    train_other_three=pickle.load(file1)
    train_other_three=train_other_three.reset_index(drop=True)
    file2=open('dfTest_trickFea.pkl','rb')
    test_other_three=pickle.load(file2)
    test_other_three=test_other_three.reset_index(drop=True)
    final_train=pd.concat([final_train,train_other_three],axis=1)
    final_test=pd.concat([final_test,test_other_three],axis=1)    
    
    df_train=df[(df.clickday>=26)&(df.clickday<29)]
    df_test=df[df.clickday==31]
    final_train.fillna(0)
    final_test.fillna(0)
    seeds=[9,99,499,999]
    
    proba_test=0
    for seed in seeds:
        X_train,X_test,y_train,y_test=train_test_split(final_train,df_train.label,
                                              test_size=0.2,random_state=0)
        lgb_train=lgb.Dataset(X_train,y_train)
        lgb_eval=lgb.Dataset(X_test,y_test,reference=lgb_train)
        params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': {'binary_logloss'},
                    'num_leaves': 63,
                        'num_trees': 100,
                        'learning_rate': 0.01,
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.8,'bagging_freq': 5
                        ,'verbose': 0,'seed':seed}


    #plst += [('eval_metric', 'logloss')]
        gbm=lgb.train(params,lgb_train,num_boost_round=1000,valid_sets=lgb_eval)

        

        proba_test_one=gbm.predict(final_test)
        proba_test+=proba_test_one
    proba_test=proba_test/4
    # submission
    df = pd.DataFrame({"instanceID": df_test["instanceID"].values.astype(int), "prob": proba_test})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
    