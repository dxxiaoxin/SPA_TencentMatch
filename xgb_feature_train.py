# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:38:23 2017

@author: Administrator
"""

import pickle
import pandas as pd
import xgboost as xgb
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
    
def stack1(final_train,final_test,df_train,df_test,params,num_round):
    X_val1=final_train[:part_len]
    y_val1=df_train.label[:part_len]
    
    X_train1=final_train.drop(X_val1.index.values)
    y_train1=df_train.label.drop(y_val1.index.values)
    
    xgb_train1=xgb.DMatrix(X_train1.values,y_train1.values)
    xgb_eval1=xgb.DMatrix(X_val1.values,y_val1.values)


    evallist1 = [(xgb_eval1, 'eval'), (xgb_train1, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst1=xgb.train(dict(params),xgb_train1,num_round,evallist1)
    proba_test_one_=xgb.DMatrix(final_test.values)
    proba_test_one=bst1.predict(proba_test_one_)
    return proba_test_one
def stack2(final_train,final_test,df_train,df_test,params,num_round):
    X_val2=final_train[part_len:2*part_len]
    y_val2=df_train.label[part_len:2*part_len]
    
    X_train2=final_train.drop(X_val2.index.values)
    y_train2=df_train.label.drop(y_val2.index.values)
    
    xgb_train2=xgb.DMatrix(X_train2.values,y_train2.values)
    xgb_eval2=xgb.DMatrix(X_val2.values,y_val2.values)


    evallist2 = [(xgb_eval2, 'eval'), (xgb_train2, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst2=xgb.train(dict(params),xgb_train2,num_round,evallist2)
    proba_test_two_=xgb.DMatrix(final_test.values)
    proba_test_two=bst2.predict(proba_test_two_)
    return proba_test_two

def stack3(final_train,final_test,df_train,df_test,params,num_round):    
    X_val3=final_train[2*part_len:3*part_len]
    y_val3=df_train.label[2*part_len:3*part_len]
    
    X_train3=final_train.drop(X_val3.index.values)
    y_train3=df_train.label.drop(y_val3.index.values)
    
    xgb_train3=xgb.DMatrix(X_train3.values,y_train3.values)
    xgb_eval3=xgb.DMatrix(X_val3.values,y_val3.values)


    evallist3 = [(xgb_eval3, 'eval'), (xgb_train3, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst3=xgb.train(dict(params),xgb_train3,num_round,evallist3)
    proba_test_three_=xgb.DMatrix(final_test.values)
    proba_test_three=bst3.predict(proba_test_three_)
    return proba_test_three

    
    
    
def stack4(final_train,final_test,df_train,df_test,params,num_round):    
    X_val4=final_train[3*part_len:4*part_len]
    y_val4=df_train.label[3*part_len:4*part_len]
    
    X_train4=final_train.drop(X_val4.index.values)
    y_train4=df_train.label.drop(y_val4.index.values)
    
    xgb_train4=xgb.DMatrix(X_train4.values,y_train4.values)
    xgb_eval4=xgb.DMatrix(X_val4.values,y_val4.values)


    evallist4 = [(xgb_eval4, 'eval'), (xgb_train4, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst4=xgb.train(dict(params),xgb_train4,num_round,evallist4)
    proba_test_four_=xgb.DMatrix(final_test.values)
    proba_test_four=bst4.predict(proba_test_four_)
    return proba_test_four


def stack5(final_train,final_test,df_train,df_test,params,num_round):    
    X_val5=final_train[4*part_len:]
    y_val5=df_train.label[4*part_len:]
    
    X_train5=final_train.drop(X_val5.index.values)
    y_train5=df_train.label.drop(y_val5.index.values)
    
    xgb_train5=xgb.DMatrix(X_train5.values,y_train5.values)
    xgb_eval5=xgb.DMatrix(X_val5.values,y_val5.values)


    evallist5 = [(xgb_eval5, 'eval'), (xgb_train5, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst5=xgb.train(dict(params),xgb_train5,num_round,evallist5)
    proba_test_five_=xgb.DMatrix(final_test.values)
    proba_test_five=bst5.predict(proba_test_five_)
    return proba_test_five
if __name__=="__main__":

    final_train,final_test,df_train,df_test=final_train_test_and_merge()
    

    
    proba_test=0

    tr_te_len=len(df_train)
    part_len=tr_te_len//5
    params = { 'n_estimators': 100, 'max_depth': 10, 
        'min_child_weight': 4, 'gamma': 0.2, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eval_metric':['auc','logloss'],'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round=110

    
    proba_test_one=stack1(final_train,final_test,df_train,df_test,params,num_round)
    
    proba_test_two=stack2(final_train,final_test,df_train,df_test,params,num_round)
    
    proba_test_three=stack3(final_train,final_test,df_train,df_test,params,num_round)
    
    proba_test_four=stack4(final_train,final_test,df_train,df_test,params,num_round)
    
    proba_test_five=stack5(final_train,final_test,df_train,df_test,params,num_round)
    '''
    #stacking 1

    X_val1=final_train[:part_len]
    y_val1=df_train.label[:part_len]
    
    X_train1=final_train.drop(X_val1.index.values)
    y_train1=df_train.label.drop(y_val1.index.values)
    
    xgb_train1=xgb.DMatrix(X_train1.values,y_train1.values)
    xgb_eval1=xgb.DMatrix(X_val1.values,y_val1.values)


    evallist1 = [(xgb_eval1, 'eval'), (xgb_train1, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst1=xgb.train(dict(params),xgb_train1,num_round,evallist1)
    proba_test_one_=xgb.DMatrix(final_test.values)
    proba_test_one=bst1.predict(proba_test_one_)
    
    #stacking 2
    X_val2=final_train[part_len:2*part_len]
    y_val2=df_train.label[part_len:2*part_len]
    
    X_train2=final_train.drop(X_val2.index.values)
    y_train2=df_train.label.drop(y_val2.index.values)
    
    xgb_train2=xgb.DMatrix(X_train2.values,y_train2.values)
    xgb_eval2=xgb.DMatrix(X_val2.values,y_val2.values)


    evallist2 = [(xgb_eval2, 'eval'), (xgb_train2, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst2=xgb.train(dict(params),xgb_train2,num_round,evallist2)
    proba_test_two_=xgb.DMatrix(final_test.values)
    proba_test_two=bst2.predict(proba_test_two_)
    
    #stacking 3
    X_val3=final_train[2*part_len:3*part_len]
    y_val3=df_train.label[2*part_len:3*part_len]
    
    X_train3=final_train.drop(X_val3.index.values)
    y_train3=df_train.label.drop(y_val3.index.values)
    
    xgb_train3=xgb.DMatrix(X_train3.values,y_train3.values)
    xgb_eval3=xgb.DMatrix(X_val3.values,y_val3.values)


    evallist3 = [(xgb_eval3, 'eval'), (xgb_train3, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst3=xgb.train(dict(params),xgb_train3,num_round,evallist3)
    proba_test_three_=xgb.DMatrix(final_test.values)
    proba_test_three=bst3.predict(proba_test_three_)
    
    #stacking 4
    X_val4=final_train[3*part_len:4*part_len]
    y_val4=df_train.label[3*part_len:4*part_len]
    
    X_train4=final_train.drop(X_val4.index.values)
    y_train4=df_train.label.drop(y_val4.index.values)
    
    xgb_train4=xgb.DMatrix(X_train4.values,y_train4.values)
    xgb_eval4=xgb.DMatrix(X_val4.values,y_val4.values)


    evallist4 = [(xgb_eval4, 'eval'), (xgb_train4, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst4=xgb.train(dict(params),xgb_train4,num_round,evallist4)
    proba_test_four_=xgb.DMatrix(final_test.values)
    proba_test_four=bst4.predict(proba_test_four_)
    
    #stacking 5
    X_val5=final_train[4*part_len:]
    y_val5=df_train.label[4*part_len:]
    
    X_train5=final_train.drop(X_val5.index.values)
    y_train5=df_train.label.drop(y_val5.index.values)
    
    xgb_train5=xgb.DMatrix(X_train5.values,y_train5.values)
    xgb_eval5=xgb.DMatrix(X_val5.values,y_val5.values)


    evallist5 = [(xgb_eval5, 'eval'), (xgb_train5, 'train')]
    #plst += [('eval_metric', 'logloss')]
    bst5=xgb.train(dict(params),xgb_train5,num_round,evallist5)
    proba_test_five_=xgb.DMatrix(final_test.values)
    proba_test_five=bst5.predict(proba_test_five_)
    '''
        




        

    proba_test=proba_test_one+proba_test_two+proba_test_three+proba_test_four+proba_test_five
    proba_test=proba_test/5
    # submission
    df = pd.DataFrame({"instanceID": df_test["instanceID"].values.astype(int), "prob": proba_test})
    #calibration
    df.prob=df.prob*1.05
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
