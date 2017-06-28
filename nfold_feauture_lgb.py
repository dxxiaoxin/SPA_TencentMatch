# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:47:38 2017

@author: Administrator
"""

import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
import numpy as np
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
    final_train.fillna(0.)
    final_test.fillna(0.)

    
    proba_test=0

    tr_te_len=len(df_train)
    part_len=tr_te_len//5
    params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': {'binary_logloss'}}
    num_round=110
    #stacking 1
    X_val1=final_train[:part_len]
    y_val1=df_train.label[:part_len]
    
    X_train1=final_train.drop(X_val1.index.values)
    y_train1=df_train.label.drop(y_val1.index.values)
    
    lgb_train1=lgb.Dataset(X_train1,y_train1)
    lgb_eval1=lgb.Dataset(X_val1,y_val1,reference=lgb_train1)



    #plst += [('eval_metric', 'logloss')]
    gbm1=lgb.train(params,lgb_train1,num_boost_round=num_round,valid_sets=lgb_eval1)
    proba_test_one=gbm1.predict(final_test)
    
    
    #stacking 2
    X_val2=final_train[part_len:2*part_len]
    y_val2=df_train.label[part_len:2*part_len]
    
    X_train2=final_train.drop(X_val2.index.values)
    y_train2=df_train.label.drop(y_val2.index.values)
    
    lgb_train2=lgb.Dataset(X_train2,y_train2)
    lgb_eval2=lgb.Dataset(X_val2,y_val2,reference=lgb_train2)

    gbm2=lgb.train(params,lgb_train2,num_boost_round=num_round,valid_sets=lgb_eval2)
    proba_test_two=gbm2.predict(final_test)
    
    #stacking 3
    X_val3=final_train[2*part_len:3*part_len]
    y_val3=df_train.label[2*part_len:3*part_len]

    X_train3=final_train.drop(X_val3.index.values)
    y_train3=df_train.label.drop(y_val3.index.values)
    
    lgb_train3=lgb.Dataset(X_train3,y_train3)
    lgb_eval3=lgb.Dataset(X_val3,y_val3,reference=lgb_train3)
 
    gbm3=lgb.train(params,lgb_train3,num_boost_round=num_round,valid_sets=lgb_eval3)
    proba_test_three=gbm3.predict(final_test)
    
    #stacking 4
    X_val4=final_train[3*part_len:4*part_len]
    y_val4=df_train.label[3*part_len:4*part_len]

    X_train4=final_train.drop(X_val4.index.values)
    y_train4=df_train.label.drop(y_val4.index.values)
    
    lgb_train4=lgb.Dataset(X_train4,y_train4)
    lgb_eval4=lgb.Dataset(X_val4,y_val4,reference=lgb_train4)

    gbm4=lgb.train(params,lgb_train4,num_boost_round=num_round,valid_sets=lgb_eval4)
    proba_test_four=gbm4.predict(final_test)    
    
    
    X_val5=final_train[4*part_len:]
    y_val5=df_train.label[4*part_len:]
    
    X_train5=final_train.drop(X_val5.index.values)
    y_train5=df_train.label.drop(y_val5.index.values)
    
    lgb_train5=lgb.Dataset(X_train5,y_train5)
    lgb_eval5=lgb.Dataset(X_val5,y_val5,reference=lgb_train5)

    gbm5=lgb.train(params,lgb_train5,num_boost_round=num_round,valid_sets=lgb_eval5)
    proba_test_five=gbm5.predict(final_test)    
   
        




        

    proba_test=proba_test_one+proba_test_two+proba_test_three+proba_test_four+proba_test_five
    proba_test=proba_test/5
    # submission
    df = pd.DataFrame({"instanceID": df_test["instanceID"].values.astype(int), "prob": proba_test})
    #calibration
    df.prob=df.prob*1.05
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
    