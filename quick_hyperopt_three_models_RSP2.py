# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:35:17 2021

@author: Administrator
"""
#import required packages
from lightgbm import LGBMRegressor as lgbm
from xgboost import XGBRegressor as xgb
import lightgbm as lgb

import gc
from hyperopt import hp, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestRegressor as rf
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
#optional but advised
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import sklearn
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

# In[1] define quick_hyperopt function
 
#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 1000 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 100 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 100 #maximum tree depth for LightGBM


#XGBOOST PARAMETERS
XGB_MAX_LEAVES = 2**12 #maximum number of leaves when using histogram splitting
XGB_MAX_DEPTH = 25 #maximum tree depth for XGBoost




#OPTIONAL OUTPUT
BEST_SCORE = 0

cv = 5

def quick_hyperopt(data, labels, package='lgbm', num_evals=NUM_EVALS, diagnostic=False):
    
    #==========
    #LightGBM
    #==========
    if package=='lgbm':
        
        print('Running {} rounds of LightGBM parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        integer_params=['n_estimators','n_estimators','max_depth','num_leaves']
       # integer_params=['learning_rate','n_estimators','max_depth','num_leaves']
        def objective(space_params):
            
            #cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            clf = lgbm(**space_params)
            cv_results=cross_val_score(clf, data, labels, cv=5,scoring='neg_mean_absolute_error', error_score='raise')
            best_loss=(-cv_results).mean()

            return{'loss':best_loss, 'status': STATUS_OK }  
        
        
        boosting_type_list=['gbdt','dart','goss']
        learning_rate_list=[0.01, 0.05, 0.07, 0.1, 0.2]
        space ={
             'boosting_type' : hp.choice('boosting_type', boosting_type_list),
                'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),
                 'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),
                'learning_rate' : hp.choice('learning_rate',learning_rate_list),
                'n_estimators': hp.quniform('n_estimators', 25, 500, 25)
            }
        

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
                
        #fmin() will return the index of values chosen from the lists/arrays in 'space'
        #to obtain actual values, index values are used to subset the original lists/arrays

        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        ##返回来的是位置
        best['boosting_type'] = boosting_type_list[best['boosting_type']]
        best['learning_rate'] = learning_rate_list[best['learning_rate']]
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        if diagnostic:
            return(best, trials)
        else:
            return(best)    

    
    #==========
    #XGBoost
    #==========
    
    elif package=='xgb':
        
        print('Running {} rounds of XGBoost parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth','n_estimators',]
      
        ##Objective Function
        def objective(space_params):
            
            for param in integer_params:
                space_params[param] = int(space_params[param])
                
            ##build models
            
            clf = xgb(**space_params)
            cv_results=cross_val_score(clf, data, labels, cv=cv,scoring='neg_mean_absolute_error', error_score='raise')
            best_loss=(-cv_results).mean()

            return{'loss':best_loss, 'status': STATUS_OK }  
        
        learning_rate_list=[0.01, 0.05, 0.07, 0.1, 0.2]
        space ={'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
                'gamma' : hp.uniform('gamma', 0, 5),
                'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
                'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1, 0.01),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 5),
                'reg_lambda' : hp.uniform('reg_lambda', 0, 5),
                'learning_rate': hp.choice('learning_rate',learning_rate_list),
                'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),
            }
        
        trials = Trials()
        ##Optimization Algorithm
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,##算法选择
                    max_evals=num_evals,  #迭代次数
                    trials=trials)##记录中间结果
        for param in integer_params:
            best[param] = int(best[param])
        best['learning_rate'] = learning_rate_list[best['learning_rate']]
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    #==========
    #RF
    #==========   
    elif package=='rf':
        print('Running {} rounds of rf parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['max_depth','max_features','n_estimators']#,'n_jobs','min_samples_leaf','min_samples_split'
      
        ##Objective Function
        def objective(space_params):
            
            for param in integer_params:
                space_params[param] = int(space_params[param])
            
            #for classification replace EVAL_METRIC_XGB_REG with EVAL_METRIC_XGB_CLASS
            clf = rf(**space_params)
            cv_results=cross_val_score(clf, data, labels, cv=cv,scoring='neg_mean_absolute_error', error_score='raise')
            best_loss=(-cv_results).mean()

            return{'loss':best_loss, 'status': STATUS_OK }
        

        ##domain space
        space ={
                'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
                'max_depth': hp.quniform('max_depth', 10, 100, 1),
               # 'n_jobs':hp.choice('n_jobs', [-1,1]),
               # 'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
              #  'min_samples_split':hp.uniform('min_samples_split',2,6),
                'max_features': hp.choice('max_features', range(1, 100))
                }

        
        trials = Trials()
        ##Optimization Algorithm
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,##算法选择
                    max_evals=num_evals,  #迭代次数
                    trials=trials)##记录中间结果
        
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)        
    #==========   
    ##LASSO
    #==========   
    elif package=='LASSO':
        print('Running {} rounds of LASSO parameter optimisation:'.format(num_evals))
        #clear space
        gc.collect()
        
        integer_params = ['alpha','max_iter']#,'n_jobs','min_samples_leaf','min_samples_split'
      
        ##Objective Function
        def objective(space_params):
            
            for param in integer_params:
                space_params[param] = int(space_params[param])
            
            #for classification replace EVAL_METRIC_XGB_REG with EVAL_METRIC_XGB_CLASS
            clf = Lasso(**space_params)
            cv_results=cross_val_score(clf, data, labels, cv=cv,scoring='neg_mean_absolute_error', error_score='raise')
            best_loss=(-cv_results).mean()

            return{'loss':best_loss, 'status': STATUS_OK }
        

        ##domain space
        alpha_list=[0.001,0.1,1,10]
        max_iter_list=[10,100,500,1000]
        space ={
                'alpha': hp.choice('alpha',alpha_list),
                'max_iter': hp.choice('max_iter',max_iter_list),
                }

        
        trials = Trials()
        ##Optimization Algorithm
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,##算法选择
                    max_evals=num_evals,  #迭代次数
                    trials=trials)##记录中间结果
        
        
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        best['alpha'] = alpha_list[best['alpha']]
        best['max_iter'] = max_iter_list[best['max_iter']]
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        
        if diagnostic:
            return(best, trials)
        else:
            return(best)        
    else:
        
        print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "LASSO" for LASSO,"rf" for random forest.')    
# In[select feature]
##创建目录
def select_feature(data,package):
    if package=='lgbm':
        model=lgbm()
    elif package=='xgb':
        model=xgb()
    elif package=='rf':
        model=rf()
    else: 
        model=lgbm()
    selector = RFECV(model, step=1, cv=5)
    selector.fit(data.drop(['Yield','ID'],axis=1),data['Yield'])
    feature = features[selector.support_]
    return feature
# In[prepare data]
    
##保存每个模型的最优参数
def save_optimal_model(data, train_labels, package, num_evals):
    if package=='lgbm':
        params = quick_hyperopt(data.values, train_labels.values, 'lgbm', num_evals)
    elif package=='xgb':
        params = quick_hyperopt(data.values, train_labels.values, 'xgb', num_evals)  
    elif package=='rf':
        params = quick_hyperopt(data.values, train_labels.values, 'rf', num_evals)
    elif package=='LASSO':
        params = quick_hyperopt(data.values, train_labels.values, 'LASSO', num_evals)
    else:
        print('Please input right model')
    return params  #np.save('rf_params.npy', rf_params)
##如果是1982-2008年建模，但是2009-2015年验证可以用这个，但是一般这种情况结果都不是很好  
# def fourmodels(train_features_raw,train_labels_raw,test_features_raw,test_labels_raw,package,params):
#     if package=='xgb':
#         train = xgb.DMatrix(train_features_raw, train_labels_raw)
#         model=xgb.train(params, train)
#         test = xgb.DMatrix(test_features_raw,test_labels_raw)
#         pred_test_raw = model.predict(test)
#         pred_train_raw = model.predict(train)
#     if package=='lgbm':
#         train = lgbm.Dataset(train_features_raw, train_labels_raw)
#         model=lgbm.train(params, train)
#         test = lgbm.Dataset(test_features_raw,test_labels_raw)
#         pred_test_raw = model.predict(test)
#         pred_train_raw = model.predict(train)
#     if package=='rf':##用cv得到训练集
#         model = rf(**params)
#         model.fit(train_features_raw,train_labels_raw)
#         pred_test_raw = model.predict(test_features_raw)
#         pred_train_raw = model.predict(train_features_raw)
#     if package=='LASSO':
#         model=Lasso(**params)
#         model.fit(train_features_raw,train_labels_raw)
#         pred_test_raw = model.predict(test_features_raw)
#         pred_train_raw = model.predict(train_features_raw)
#     else:
#          print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "LASSO" for LASSO,"rf" for random forest.') 
         
#     return pred_test_raw, pred_train_raw


##R2计算的结果不一样excel计算的要大些，因为原理不同

from scipy import stats
def rsquared(x, y): 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    return r_value


def accuracy(ori_data,pre_data):
    evalation_indices=pd.DataFrame(columns=['test_mae','test_rmse','test_r2','test_r2_adjust'])
    evalation_indices.loc[0,'test_mae']=mean_absolute_error(ori_data,pre_data)
    evalation_indices.loc[0,'test_rmse']=np.sqrt(mean_squared_error(ori_data,pre_data))
    evalation_indices.loc[0,'test_r2']=r2_score(ori_data,pre_data)
    evalation_indices.loc[0,'test_r2_adjust']=rsquared(ori_data,pre_data)
    return evalation_indices


def cross_train(feature_raw,lables_raw,package):
    if package=='xgb':
        model = xgb()
        #model = xgb(**params)
        scores_rmse = np.sqrt(-cross_val_score(model, feature_raw,lables_raw, scoring='neg_mean_squared_error',cv=cv, n_jobs=-1))
        scores_r2 = cross_val_score(model, feature_raw,lables_raw, scoring='r2',cv=cv, n_jobs=-1) ##仅仅用目标集标签的平均值，就能让R2_score为0
        scores_mae = -cross_val_score(model, feature_raw,lables_raw, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        
    elif package=='lgbm':
        model = lgbm()
        #model = lgbm(**params)
        scores_rmse = np.sqrt(-cross_val_score(model, feature_raw,lables_raw, scoring='neg_mean_squared_error',cv=cv, n_jobs=-1))
        scores_r2 = cross_val_score(model, feature_raw,lables_raw, scoring='r2',cv=cv, n_jobs=-1) ##仅仅用目标集标签的平均值，就能让R2_score为0
        scores_mae = -cross_val_score(model, feature_raw,lables_raw, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        
    elif package=='rf':##用cv得到训练集
        model = rf()
        #model = rf(**params)
        scores_rmse = np.sqrt(-cross_val_score(model, feature_raw,lables_raw, scoring='neg_mean_squared_error',cv=cv, n_jobs=-1))
        scores_r2 = cross_val_score(model, feature_raw,lables_raw, scoring='r2',cv=cv, n_jobs=-1) ##仅仅用目标集标签的平均值，就能让R2_score为0
        scores_mae = -cross_val_score(model, feature_raw,lables_raw, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        
    elif package=='LASSO':
        model=Lasso()
       # model=Lasso(**params)
        scores_rmse = np.sqrt(-cross_val_score(model, feature_raw,lables_raw, scoring='neg_mean_squared_error',cv=cv, n_jobs=-1))
        scores_r2 = cross_val_score(model, feature_raw,lables_raw, scoring='r2',cv=cv, n_jobs=-1) ##仅仅用目标集标签的平均值，就能让R2_score为0
        scores_mae = -cross_val_score(model, feature_raw,lables_raw, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    else:
          print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "LASSO" for LASSO,"rf" for random forest.') 
   
    ##R2小于0表示随机模型的值都比他这个好，因此设置模型为null##返回的是一个list
    indexs=pd.DataFrame(columns=['train_mae','train_rmse','train_r2'])
    indexs.loc[0,'train_mae'],indexs.loc[0,'train_rmse'] ,indexs.loc[0,'train_r2']= scores_mae.mean(), scores_rmse.mean(),scores_r2.mean()         
    return indexs ##返回均值方差和预测值
##LASSO是输出的是系数
def test_results(train_feature_raw,train_lables_raw,test_feature_raw,test_lables_raw,package):
    
    if package=='xgb':
       #  model = xgb(**params)
        model = xgb()
        model_train=model.fit(train_feature_raw,train_lables_raw)
        feat_imp = pd.DataFrame({'column': train_feature_raw.columns, 'importance': model_train.feature_importances_}).sort_values(by='importance',ascending=False)
        predic_labels_raw=model_train.predict(test_feature_raw)
        
    elif package=='lgbm':
        model = lgbm()
        #model = lgbm(**params)
        model_train=model.fit(train_feature_raw,train_lables_raw)
        predic_labels_raw=model_train.predict(test_feature_raw)
        feat_imp = pd.DataFrame({'column': train_feature_raw.columns, 'importance': model_train.feature_importances_}).sort_values(by='importance',ascending=False)
    elif package=='rf':##用cv得到训练集
        ##model = rf(**params)
        model = rf()
        model_train=model.fit(train_feature_raw,train_lables_raw)
        predic_labels_raw=model_train.predict(test_feature_raw)
        feat_imp = pd.DataFrame({'column': train_feature_raw.columns, 'importance': model_train.feature_importances_}).sort_values(by='importance',ascending=False)
    elif package=='LASSO':
        model=Lasso()
        #model=Lasso(**params)
        model_train=model.fit(train_feature_raw,train_lables_raw)
        predic_labels_raw=model_train.predict(test_feature_raw)
        feat_imp = pd.DataFrame({'column': train_feature_raw.columns, 'importance': model_train.coef_}).sort_values(by='importance',ascending=False)
    else:
          print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "LASSO" for LASSO,"rf" for random forest.') 
          
    test_result=accuracy(predic_labels_raw,test_lables_raw)
    return test_result,predic_labels_raw,feat_imp
##accuray evaluation function
# package='xgb'
# num_evals=10

##按照1982-2008训练，2009-2015年验证
#     train_features_raw,train_labels_raw=data[data['year']<=2008].drop(['Yield','year'],axis=1),data[data['year']<=2008]['Yield']
#     test_features_raw,test_labels_raw=data[data['year']>2008].drop(['Yield','year'],axis=1),data[data['year']>2008]['Yield']
#      model_train=model.fit(train_features_raw,train_labels_raw)
 #     predic_labels_raw=model_train.predict(test_features_raw)
 
def index_year(labels_two_RSP):
    evalation_indices=pd.DataFrame(columns=['year','test_rmse','test_r2','test_r2_adjust'])
    for year in labels_two_RSP['year'].unique():
        data_one=labels_two_RSP[labels_two_RSP['year']==year]
        evalation_indice=accuracy(data_one['lables_raw'],data_one['predic_raw'])
        evalation_indice['year']=year
        evalation_indices=evalation_indices.append(evalation_indice,ignore_index=True)     
        print(evalation_indices)
    return evalation_indices

 ##测试一组数据 data_one=data[data['ID']==data['ID'].unique()[1]]
 
# =============================================================================
#  如果Type==1，就是分区域建立
#  如果Type=2，就是分ID建立模型
#  最后输出的数据应该只有留一手法的test元数据和重要性和计算的指标
# =============================================================================
def result_RSP(data,package,num_evals):
    evalation_indices=pd.DataFrame(columns=['train_mae','train_rmse','train_r2','test_mae','test_rmse','test_r2','test_r2_adjust'])
    labels_two_RSP=pd.DataFrame(columns=['lables_raw','predic_raw','ID','year'])
    feat_imps=pd.DataFrame()
    
    
    ##分区的话，留一手法
    for year in data['year'].unique():
        data_train=data[data['year']!=year]
        data_test=data[data['year']==year]
        feature=select_feature(data,package)    ##数据特征选择
        train_feature_raw,train_lables_raw=data_train.drop(['Yield','ID'],axis=1)[feature],data_train['Yield']
        test_feature_raw,test_lables_raw=data_test.drop(['Yield','ID'],axis=1)[feature],data_test['Yield']
        
        ##查补数据
        train_feature_raw,train_lables_raw=train_feature_raw.fillna(method='ffill',axis=0),train_lables_raw.fillna(method='ffill',axis=0)
        test_feature_raw,test_lables_raw=test_feature_raw.fillna(method='ffill',axis=0),test_lables_raw.fillna(method='ffill',axis=0)
        
        #params=save_optimal_model(train_feature_raw,train_lables_raw, package, num_evals)
        indexs_train=cross_train(train_feature_raw,train_lables_raw,package)
        indexs_test,predic_labels_raw,feat_imp=test_results(train_feature_raw,train_lables_raw,test_feature_raw,test_lables_raw,package)
        index=pd.concat([indexs_train,indexs_test],axis=1) ##按照行合并
        index['year'],feat_imp['year']=year,year
        labels_two=pd.DataFrame(columns=['lables_raw','predic_raw','year','ID'])
        labels_two['lables_raw'],labels_two['predic_raw'],labels_two['ID'],labels_two['year']=data_test['Yield'],predic_labels_raw,data_test['ID'],data_test['year']
        labels_two_RSP=labels_two_RSP.append(labels_two,ignore_index=True) 
        feat_imps=feat_imps.append(feat_imp)
        evalation_indices=evalation_indices.append(index,ignore_index=True)                
     ####我可能吃多了，才计算这一步骤           
    Index_year=index_year(labels_two_RSP)
    evalation_indices.iloc[:,-4:-1]=Index_year.iloc[:,1:]
    return evalation_indices,labels_two_RSP,feat_imps

    ##不能一个一个单元格去建立，否则会有错误；
    # for ID in data['ID'].unique():
    #     data_one=data[data['ID']==ID]
    #     feature_raw,lables_raw=data_one.drop(['Yield','ID'],axis=1),data_one['Yield']
    #     params=save_optimal_model(feature_raw,lables_raw, package, num_evals)
    #     indexs,predic_raw=fourmodels(feature_raw,lables_raw,params,package)
    #     indexs['ID']=ID
    #     labels_two=pd.DataFrame(columns=['lables_raw','predic_raw','year','ID'])
    #     labels_two['lables_raw'],labels_two['predic_raw'],labels_two['ID']=lables_raw,predic_raw,ID
    #     labels_two.loc[:,'year']=list(range(2000,2016,1))
    #     print(indexs)
        # evalation_indices=evalation_indices.append(indexs,ignore_index=True)
        # labels_two_RSP=labels_two_RSP.append(labels_two,ignore_index=True)    




def index_polit(labels_two_RSP):
    evalation_indices=pd.DataFrame(columns=['ID','test_rmse','test_r2','test_r2_adjust'])
    for ID in labels_two_RSP['ID'].unique():
        data_one=labels_two_RSP[labels_two_RSP['ID']==ID]
        evalation_indice=accuracy(data_one['lables_raw'],data_one['predic_raw'])
        evalation_indice['ID']=ID
        evalation_indices=evalation_indices.append(evalation_indice,ignore_index=True)
        print(evalation_indices)
    return evalation_indices
# In[prepare data]
# #from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import LeaveOneOut
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import pandas as pd

##cross_val_predict 输出结果
##cross_val_score  输出R2
# def cross_model(data,package):
#     feature_raw,lables_raw=data.drop(['Yield','ID'],axis=1),data['Yield']
#     params=save_optimal_model(feature_raw, lables_raw, package, 50)
#     cv = LeaveOneOut()
    
#     scores = cross_val_score(model, data.drop(['Yield','ID'],axis=1),data['Yield'],scoring='neg_mean_squared_error',
#                          cv=cv, n_jobs=-1)
    
    

# def results_model(data,package):
#     ##split train and test data
#     train_features_raw,train_labels_raw=data[data['year']<=2008].drop('Yield',axis=1),data[data['year']<=2008]['Yield']
#     test_features_raw,test_labels_raw=data[data['year']>2008].drop('Yield',axis=1),data[data['year']>2008]['Yield']
#     params=save_optimal_model(train_features_raw, train_labels_raw, package, 50)
#   # np.save('rf_params.npy', params)##前面加路径
#     pred_test_raw, pred_train_raw=fourmodels(train_features_raw,train_labels_raw,test_features_raw,test_labels_raw,package,params)
#     train_accuray=accuracy(train_labels_raw,pred_train_raw)
#     test_accuray=accuracy(test_labels_raw,pred_test_raw)
#     return params, train_accuray, test_accuray



##train scaler##树模型不需要归一化，所以保持
# train_features_scaler = StandardScaler().fit_transform(train_features_raw)
# scaler_train = StandardScaler().fit(train_labels_raw.values.reshape(-1, 1))
# train_labels_scaler = scaler_train.transform(train_labels_raw.values.reshape(-1, 1))
# ##test scaler
# test_features_scaler = StandardScaler().fit_transform(test_features_raw)
# scaler_test= StandardScaler().fit(test_features_raw.values.reshape(-1, 1))
# test_labels_scaler = scaler_test.transform(test_labels_raw.values.reshape(-1, 1))
##return original

##pred_test_raw=scaler_test_lablel.inverse_transform(pred_test_raw)


# In[1] output parameter results

