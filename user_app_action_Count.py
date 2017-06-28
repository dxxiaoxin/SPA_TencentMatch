# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:30:23 2017

@author: Administrator
"""

import pandas as pd
import pickle 


def user_action_Count():
    user_installedapps=pd.read_csv('./final/user_app_actions.csv')
    #user_history_installed=pd.read_csv('./final/user_installedapps.csv')
    app_categories=pd.read_csv('./final/app_categories.csv')
    user_installedapps['installday']=user_installedapps.installTime.map(lambda x:int(str(x)[:2]))
    user_installedapps=user_installedapps[user_installedapps.installday<27]
    user_installedapps=pd.merge(user_installedapps,app_categories,how='left',on='appID')

    #用户共安装了多少app
    user_installedapps['user_installed_app_Count']=1
    u1=user_installedapps.ix[:,['userID','user_installed_app_Count']].groupby('userID').agg('sum').reset_index()
    
    #用户共安装多少类型的app
    user_installedapps['user_installed_cate_Count']=1
    u2=user_installedapps.ix[:,['userID','appCategory','user_installed_cate_Count'
                                ]].groupby(['userID','appCategory']).agg('sum').reset_index()
   # user_installedapps=pd.merge(user_installedapps,u1,how='left',on='userID')
    #user_installedapps=pd.merge(user_installedapps,u2,how='left',on=['userID','appCategory'])
    user_installedapps['appID_installed_Count']=1
    u3=user_installedapps.ix[:,['appID','appID_installed_Count']].groupby('appID').agg('sum').reset_index()
    #user_installedapps=pd.merge(user_installedapps,u3,how='left',on='appID')
    
    #user_history_installed['user_history_installed_Count']=1
    #u4=user_history_installed.ix[:,['userID','user_history_installed_Count']].groupby('userID').agg('sum').reset_index()
    
    #user_history_installed['appID_history_installed_Count']=1
    #u5=user_history_installed.ix[:,['appID','appID_history_installed_Count']].groupby('appID').agg('sum').reset_index()    
    #user_installedapps=user_installedapps[['userID','user_installed_app_Count','user_installed_cate_Count','appID_installed_Count']]
    output1 = open('user_installedapps_Counts.pkl', 'wb')
    pickle.dump(u1, output1)
    output2 = open('user__cate_installedapps_Counts.pkl', 'wb')
    pickle.dump(u2, output2)
    output3 = open('appID_installedapps_Counts.pkl', 'wb')
    pickle.dump(u3, output3)
    '''
    output4 = open('user_history_installed_Counts.pkl', 'wb')
    pickle.dump(u4, output4)
    output5 = open('appID_history_installed_Counts.pkl', 'wb')
    pickle.dump(u5, output5)
    '''
    return u1,u2,u3
if __name__=="__main__":
    u1,u2,u3=user_action_Count()