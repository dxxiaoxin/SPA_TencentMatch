# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 21:41:34 2017

@author: Administrator
"""

import pandas as pd
import pickle
def convert_age(age_str):
    if age_str==0:
        return 0
    elif 0<age_str<=6:
        return 1
    elif 6<age_str<=18:
        return 2
    elif 18<age_str<=24:
        return 3
    elif 24<age_str<=30:
        return 4
    elif 30<age_str<=40:
        return 5
    elif 40<age_str<=50:
        return 6
    elif age_str>55:
        return 7
    else:
        return 0
def convert_time(hour_str):
    if 0<=hour_str<=4:
        return 5
    elif 5<=hour_str<=10:
        return 1
    elif 11<=hour_str<=13:
        return 2
    elif 14<=hour_str<=19:
        return 3
    elif 20<=hour_str<24:
        return 4
    else:
        return 0
def click_convert(df):
    #时间划分
    df['clickday']=df.clickTime.map(lambda x:int(str(x)[:2]))
    df['clickhour']=df.clickTime.map(lambda x:int(str(x)[2:4]))
    return df
    
def appCate_split(df):
    df['appCate_Max']=df.appCategory.map(lambda x:int(str(x)[0]))
    return df
def userID_represent_age_gender_education(df):
    df['age_gender_education_marriageStatus']=df.age.map(lambda x:str(x))+df.gender.map(lambda x:str(x))+ df.education.map(lambda x:str(x))+df.marriageStatus.map(lambda x:str(x)) 
    df['advertiserID_appCate_Max_appPlatform']=df.advertiserID.map(lambda x:str(x))+df.appCate_Max.map(lambda x:str(x))+ df.appPlatform.map(lambda x:str(x))
    return df
    
    
    
    
def train_test_concat_and_merge_other_list():
    train=pd.read_csv('./final/train.csv')
    test=pd.read_csv('./final/test.csv')
    df=pd.concat([train,test])
    
    #other lists
    ad=pd.read_csv('./final/ad.csv')
    app_categories=pd.read_csv('./final/app_categories.csv')
    ad=pd.merge(ad,app_categories,how='left',on='appID')
    user=pd.read_csv('./final/user.csv')
    position=pd.read_csv('./final/position.csv')    
    df=pd.merge(df,ad,how='left',on='creativeID')
    df=pd.merge(df,user,how='left',on='userID')
    df=pd.merge(df,position,how='left',on='positionID')
    df=click_convert(df)
    df=appCate_split(df)
    df=userID_represent_age_gender_education(df)
    
    df.age=df.age.map(convert_age)
    df.clickhour=df.clickhour.map(convert_time)
    output = open('train_test_concat_and_merge.pkl', 'wb')
    pickle.dump(df, output)
    return df
    
if __name__=="__main__":
    df=train_test_concat_and_merge_other_list()    
    
    
