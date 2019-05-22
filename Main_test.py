import pandas as pd
import numpy as np
import glob
import datetime
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import pickle
import datetime as dt

from Fonctions_creation_dataset import f_all_data
#from Fonctions_creation_dataset import f_all_data_0
from Fonctions_creation_dataset import get_data_esm_0
from Fonctions_creation_dataset import get_data_rad
from Fonctions_creation_dataset import prod_data
from clean_data import data_clean                                                                            
from clean_data import prod_clean
from clean_data import prod_clean_0
from clean_data import prod_clean_rad
from clean_data import prod_data_del
from clean_data import prod_data_mean
from clean_data import prod_data_median
from clean_data import rad_mask
from clean_data import night_mask
from clean_data import night_mask_ind
from clean_data import premier_nettoyage_rad
from clean_data import test_rad
from clean_data import rad_nan
from clean_data import rad_hour
from clean_data import clean_owm
#from Fonctions_creation_dataset import data_rad
from Fonctions_creation_dataset import hour_mean
from Fonctions_creation_dataset import all_data_rad
from Fonctions_creation_dataset import test_data_rad
from Fonctions_creation_dataset import all_data_rad_num
from Fonctions_creation_dataset import all_data_rad_dummies
from Fonctions_creation_dataset import get_data_owm_forecast
from Fonctions_creation_dataset import get_data_owm_forecast_hourly
from Fonctions_creation_dataset import get_owm_history
from Fonctions_creation_dataset import owm_comp
from Fonctions_creation_dataset import data_dummies
from Fonctions_creation_dataset import data_num
from Fonctions_creation_dataset import all_data_rad_dummies_weather
from Fonctions_creation_dataset import rad_pred_val
from Fonctions_creation_dataset import rad_pred_prev
from Fonctions_creation_dataset import X_OWM_3
from Fonctions_creation_dataset import X_OWM_1
from Fonctions_creation_dataset import comp_rad
from Machine_learning import result_grid_search
from Machine_learning import result_grid_search_minmax
from Machine_learning import error_min_max
from Machine_learning import report_grid_search
from Machine_learning import results_random_forest
from Machine_learning import results_random_forest_test
from Machine_learning import results_random_forest_minmax
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
#%%

owm_forecast=get_data_owm_forecast()
owm_forecast_hour=get_data_owm_forecast_hourly()
owm_forecast_comp=owm_comp()
owm_history=get_owm_history()
owm_history=clean_owm(owm_history)
[owm_indna,owm_datena]=test_rad(owm_history)
rad=get_data_rad()
#rad_n=rad_nan(rad)
#rad_m=rad_mask(rad,'Date')
rad_h=rad_hour(rad)
data_rad=all_data_rad() 
#[ind_na,dates_na]=test_rad(rad_m)
#rad_hour=hour_mean(rad_m)
#data_r=data_rad(rad_hour)
#data_rad=premier_nettoyage_rad(data_rad)
data_rad_num=all_data_rad_num()
#data_rad_num=data_rad_num.drop(['day','day_number','humidity'],axis=1)
#
#data_rad_num=data_rad_num.set_index('Date')

data_rad_dum=all_data_rad_dummies()
#data_rad_dum=data_rad_dum.drop(['day','day_number','humidity'],axis=1)
#data_rad_dum=data_rad_dum.set_index('Date')

data_rad_dumw=all_data_rad_dummies_weather()
#data_rad_dumw=data_rad_dumw.drop(['day','day_number','humidity'],axis=1)
#data_rad_dumw=data_rad_dumw.set_index('Date')
#data_rad_dum.to_csv('data_rad_dum.csv', sep=";", decimal=",")
#all data useful for the project 
#esm_0=get_data_esm_0()
#esm_0=prod_clean_0(esm_0)
#²all_data_0=f_all_data_0()
#all_data=data_clean(all_data)
#all_data['Solar_Rad - W/m2']=all_data['Solar_Rad - W/m2']/100
#dat useful for solar production prediction
#
#p_data=prod_data()
#test_rad=test_data_rad()
#p_data_del=prod_data_del(p_data)
#p_data_mean=prod_data_mean(p_data)
#p_data_med=prod_data_median(p_data)

#m=p_data.groupby('day_number')['GisementSolaire'].idxmax()
#mh=p_data.hour[m]
#p_data=prod_clean(p_data)
#p_data.to_csv('prod_data_0.csv',index=False, sep=";", decimal=",")
#print('fichier prod_data_0.csv créé')
#p_data_rad=pd.read_csv('data_clean_radiance.csv')
#p_data_rad=prod_clean_rad(p_data_rad)

#%%
# Machine learning

# Random Forest
from Machine_learning import X_y
from Machine_learning import X_y_time
from Machine_learning import train_random_forest
from Machine_learning import train_grid_search
from Machine_learning import train_grid_search_time
from Machine_learning import rmse
from Machine_learning import train_grid_search_test
from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import random_forest_test
from sklearn.metrics import mean_squared_error
#data_rad_num=data_rad_num.drop(['day_number','humidity','pressure','month'],axis=1)
#[X_train, X_test, y_train, y_test]=X_y(data_rad_num,1)
#[X_train_d, X_test_d, y_train_d, y_test_d]=X_y(data_rad_dum,1)
#[X_train_dw, X_test_dw, y_train_dw, y_test_dw]=X_y(data_rad_dumw,1)


#frst=train_random_forest(data_rad_num,1)

#numerical values for month hour and weather
#frst_gs=train_grid_search(data_rad_num,0)

#categorical values for month hour and weather
#frst_gs_d=train_grid_search(data_rad_dum,0)

#frst_gs_dw=train_grid_search(data_rad_dumw,0)

#%% Split dataset chronologically
X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t=X_y_time(data_rad_num,0)
X_train_d_t, X_val_d_t, X_test_d_t, y_train_d_t, y_val_t, y_test_d_t=X_y_time(data_rad_dum,0)
X_train_dw_t, X_val_dw_t, X_test_dw_t, y_train_dw_t, y_val_t, y_test_dw_t=X_y_time(data_rad_dumw,0)

X_train_t_m, X_val_t_m, X_test_t_m, y_train_t_m, y_val_t_m, y_test_t_m,scaler_X_t_m,scaler_y_t_m=X_y_time(data_rad_num,1)
X_train_d_t_m, X_val_d_t_m, X_test_d_t_m, y_train_d_t_m, y_val_t_m, y_test_d_t_m,scaler_X_d_t_m,scaler_y_d_t_m=X_y_time(data_rad_dum,1)
X_train_dw_t_m, X_val_dw_t_m, X_test_dw_t_m, y_train_dw_t_m, y_val_t_m, y_test_dw_t_m,scaler_X_dw_t_m,scaler_y_dw_t_m=X_y_time(data_rad_dumw,1)
#%%
#X_prevision=owm_forecast
#X_prevision, cat=data_num(X_prevision)
#X_prevision_group=owm_forecast.resample('H')
#X_prevision=X_prevision_group.mean()
#X_prevision['weather_main']=X_prevision['weather_main'].interpolate(method='pad')
#X_prevision=X_prevision.interpolate()
#X_prevision['weather_main']=cat.take(X_prevision['weather_main']-1)
#X_prevision['hour']=X_prevision.index.hour
#X_prevision['month']=X_prevision.index.month
#X_prevision['month_name']=X_prevision.index.month_name()
#X_prevision=data_dummies(X_prevision)
#X_prevision=X_prevision.drop('humidity',axis=1)
X_prevision=X_OWM_3()
X_prevision_hour=X_OWM_1()
#%% 
reg=RandomForestRegressor(max_features='auto', min_samples_leaf=2,min_samples_split=50,n_estimators=100, random_state=0)
frst=reg.fit(X_train_t,y_train_t )

reg_get=RandomForestRegressor(max_features='auto', min_samples_leaf=1,min_samples_split=2,n_estimators=10, random_state=0)
frst_get=reg_get.fit(X_train_t,y_train_t )
#%%
#frst=train_random_forest(data_rad_num,1)

#numerical values for month hour and weather
frst_gs_t=train_grid_search_time(data_rad_num,0)
frst_gs_t_m=train_grid_search_time(data_rad_num,1)

#categorical values for month hour and weather
frst_gs_d_t=train_grid_search_time(data_rad_dum,0)
frst_gs_d_t_m=train_grid_search_time(data_rad_dum,1)

frst_gs_dw_t=train_grid_search_time(data_rad_dumw,0)
frst_gs_dw_t_m=train_grid_search_time(data_rad_dumw,1)

#%%
#score_train=frst_gs.score(X_train,y_train)
#params=frst_gs.get_params()
#params_d=frst_gs_d.get_params()
#score_test=frst_gs.score(X_test,y_test)
#
#score_train_d=frst_gs_d.score(X_train_d,y_train_d)
#params_d=frst_gs_d.get_params()
#score_test_d=frst_gs_d.score(X_test_d,y_test_d)
#
#score_train_dw=frst_gs_dw.score(X_train_dw,y_train_dw)
#params_dw=frst_gs_dw.get_params()
#score_test_dw=frst_gs_dw.score(X_test_dw,y_test_dw)
##dp=frst.decision_path(X_train)
#importance=frst.feature_importances_
#y_pred_train=frst_gs.predict(X_train)
#y_pred=frst_gs.predict(X_test)
#rmse_test=rmse(y_test,y_pred)
#
#y_pred_d_train=frst_gs_d.predict(X_train_d)
#y_pred_d=frst_gs_d.predict(X_test_d)
#rmse_test_d=rmse(y_test_d,y_pred_d)

#y_pred_dw_train=frst_gs_dw.predict(X_train_dw)
#y_pred_dw=frst_gs_dw.predict(X_test_dw)
#rmse_test_dw=rmse(y_test_dw,y_pred_dw)
#plt.figure()
#plt.plot(y=[y_test,y_pred],colormap='jet')
#plt.show()

#score_train_t=frst_gs_t.score(X_train_t,y_train_t)
#params_t=frst_gs_t.get_params()
#params_d_t=frst_gs_d_t.get_params()
#score_test_t=frst_gs_t.score(X_test_t,y_test_t)
#
#score_train_d_t=frst_gs_d_t.score(X_train_d_t,y_train_d_t)
#params_d_t=frst_gs_d_t.get_params()
#score_test_d_t=frst_gs_d_t.score(X_test_d_t,y_test_d_t)
#
#score_train_dw_t=frst_gs_dw_t.score(X_train_dw_t,y_train_dw_t)
#params_dw_t=frst_gs_dw_t.get_params()
#score_test_dw_t=frst_gs_dw_t.score(X_test_dw_t,y_test_dw_t)

#%%


means = frst_gs.cv_results_['mean_test_score']
stds = frst_gs.cv_results_['std_test_score']


means_d = frst_gs_d.cv_results_['mean_test_score']
stds_d = frst_gs_d.cv_results_['std_test_score']

means_dw = frst_gs_dw.cv_results_['mean_test_score']
stds_dw = frst_gs_dw.cv_results_['std_test_score']


means_t = frst_gs_t.cv_results_['mean_test_score']
stds_t = frst_gs_t.cv_results_['std_test_score']

means_train = frst_gs_t.cv_results_['mean_train_score']
stds_train = frst_gs_t.cv_results_['std_train_score']

results_gs=[]
i=0
for mean, std, params in zip(means_t, stds_t, frst_gs_t.cv_results_['params']):
    results_gs.append(i)
    results_gs[i]=[mean, std * 2, params]
    i+=1
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

results_gs_train=[]
i=0
for mean, std, params in zip(means_train, stds_train, frst_gs_t.cv_results_['params']):
    results_gs_train.append(i)
    results_gs_train[i]=[mean, std * 2, params]
    i+=1
    
means_d_t = frst_gs_d_t.cv_results_['mean_test_score']
stds_d_t = frst_gs_d_t.cv_results_['std_test_score']

means_dw_t = frst_gs_dw_t.cv_results_['mean_test_score']
stds_dw_t = frst_gs_dw_t.cv_results_['std_test_score']

from visualisation import all_plots
from visualisation import all_reg_pairplots
from visualisation import all_density_plots
from visualisation import prod_plots
from visualisation import prod_plots_del
from visualisation import prod_plots_mean
from visualisation import prod_plots_med
from visualisation import boxplot_by_hour
from visualisation import boxplots_del
from visualisation import boxplots_mean
from visualisation import boxplots_month
from visualisation import plot_temps
from visualisation import all_pairplots_del
from visualisation import plot_rad
from visualisation import plot_rad_hour
from visualisation import plot_columns
from visualisation import all_plot_rad
from visualisation import plot_owm
from Analyse import stats_owm
#[[mean_h,var_h,std_h],[mean_f,var_f,std_f]]=stats_owm(owm_history.loc['21-01-2019 00:00:00':'13-02-2019 00:00:00'],owm_forecast)
#
#owm=plot_owm(owm_history,owm_forecast)
#all_plot_rad(owm_h)
#all_plot_rad(owm_forecast)         
#boxplot_by_hour(rad_h,'Radiance')
#plot_rad_hour(rad_hour)
#plot_temps(p_data)
#prod_plots(esm_0)
#boxplots_del(p_data_del)
#prod_plots_mean(p_data_mean)
#prod_plots(p_data)
#boxplots_month(p_data_med)
#boxplots_mean(p_data_del)
#sns.catplot(x="hour", y="weather_main", data=p_data_del);
#all_reg_pairplots(all_data)
#all_density_plots(p_data_mean)
#all_plots(all_data)
#all_pairplots_del(p_data_del)
#%% Predictions
y_pred_train_t=frst_gs_t.predict(X_train_t)
y_pred_train_d_t=frst_gs_d_t.predict(X_train_d_t)
y_pred_train_dw_t=frst_gs_dw_t.predict(X_train_dw_t)
y_pred_train_test=frst.predict(X_train_t)
y_pred_train_get=frst_get.predict(X_train_t)

#%%

y_pred_val_t=frst_gs_t.predict(X_val_t)
y_pred_val_d_t=frst_gs_d_t.predict(X_val_d_t)
y_pred_val_dw_t=frst_gs_dw_t.predict(X_val_dw_t)
y_pred_val_test=frst.predict(X_val_t)
y_pred_val_get=frst_get.predict(X_val_t)

rmse_val_t=rmse(y_val_t,y_pred_val_t)
rmse_val_d_t=rmse(y_val_t,y_pred_val_d_t)
rmse_val_dw_t=rmse(y_train_t,y_pred_train_dw_t) 

#%%
y_pred_t=frst_gs_t.predict(X_test_t)
y_pred_d_t=frst_gs_d_t.predict(X_test_d_t)
y_pred_dw_t=frst_gs_dw_t.predict(X_test_dw_t)
y_pred_test=frst.predict(X_test_t)
y_pred_get=frst_get.predict(X_test_t)
#%%
t_train=list(range(len(y_train_t)))
t_test=list(range(len(y_test_t)))
plt.figure()
plt.plot(t_train,y_train_t,t_train,y_pred_train_t)
plt.show()

plt.figure()
plt.plot(t_train,y_train_d_t,t_train,y_pred_train_d_t)
plt.show()

plt.figure()
plt.plot(t_train,y_train_dw_t,t_train,y_pred_train_dw_t)
plt.show()
#plt.figure()
#plt.plot(t_train,y_train,t_train,y_pred_train)
#plt.show()

#%%Scores
rmse_train_t=rmse(y_train_t,y_pred_train_t)
rmse_train_d_t=rmse(y_train_t,y_pred_train_d_t)
rmse_train_dw_t=rmse(y_train_t,y_pred_train_dw_t)   
rmse_train_t=rmse(y_train_t,y_pred_train_t)
rmse_train_test=rmse(y_train_t,y_pred_train_test)
rmse_train_get=rmse(y_train_t,y_pred_train_get)

rmse_test_t=rmse(y_test_t,y_pred_t)
rmse_test_d_t=rmse(y_test_t,y_pred_d_t)
rmse_test_dw_t=rmse(y_test_t,y_pred_dw_t)
rmse_test_test=rmse(y_test_t,y_pred_test)
rmse_test_get=rmse(y_test_t,y_pred_get)

params_t=frst_gs_t.get_params()
params_d_t=frst_gs_d_t.get_params()
params_dw_t=frst_gs_dw_t.get_params()

estimator=frst_gs_t.best_estimator_

mse=mean_squared_error(y_train_t,y_pred_train_test)
#%%

gs=train_grid_search_test(data_rad_num,0)
gs_fit=gs.fit(X_train_t,y_train_t)
#%%
gs_param=gs.get_params()
gs_estim=gs.best_estimator_
gs_fit_param=gs_fit.get_params()
gs_fit_estim=gs_fit.best_estimator_

gs_best_param=frst_gs_t.best_params_
gs_fit_param=gs_fit.get_params()
gs_best_estim=frst_gs_t.best_estimator_

gs_best_param_d=frst_gs_d_t.best_params
gs_fit_param=gs_fit.get_params()
gs_best_estim_d=frst_gs_d_t.best_estimator_

gs_best_param_dw=frst_gs_dw_t.best_params_
gs_fit_param=gs_fit.get_params()
gs_best_estim_dw=frst_gs_dw_t.best_estimator_

#%%
from visualisation import train_error
from visualisation import plot_grid_search

n_estimators=[10, 50, 100, 200,300 ,500]
min_samples_split=[0.1,0.3,0.5,0.7,0.9,2,3,4,5,10,50,100]
plot_grid_search(frst_gs_t.cv_results_, n_estimators, min_samples_split, 'n estimators', 'min samples split')
plot_grid_search(frst_gs_d_t.cv_results_, n_estimators, min_samples_split, 'n estimators', 'min samples split')
plot_grid_search(frst_gs_dw_t.cv_results_, n_estimators, min_samples_split, 'n estimators', 'min samples split')
plot_grid_search(frst_gs_t.cv_results_, min_samples_split, n_estimators, 'min samples split', 'n estimators')
plot_grid_search(frst_gs_d_t.cv_results_, min_samples_split, n_estimators, 'min samples split', 'n estimators')
plot_grid_search(frst_gs_dw_t.cv_results_, min_samples_split, n_estimators, 'min samples split', 'n estimators')

#train_error(frst_gs_t,X_train_t,y_train_t)
#%%


results=result_grid_search(frst_gs_t)
results_d=result_grid_search(frst_gs_d_t)
results_dw=result_grid_search(frst_gs_dw_t)

results_m=result_grid_search_minmax(scaler_y_t_m,data_rad,frst_gs_t_m)
results_d_m=result_grid_search_minmax(scaler_y_d_t_m,data_rad,frst_gs_d_t_m)
results_dw_m=result_grid_search_minmax(scaler_y_dw_t_m,data_rad,frst_gs_dw_t_m)

#%%
report_grid_search(frst_gs_t)
report_grid_search(frst_gs_d_t)
report_grid_search(frst_gs_dw_t)
report_grid_search(frst_gs_t_m)
report_grid_search(frst_gs_d_t_m)
report_grid_search(frst_gs_dw_t_m)
#%%
y_pred_val_t_1, rmse_val_t_1=results_random_forest(data_rad_num,50,300,'frst_1_cor.p')
y_pred_val_t_2, rmse_val_t_2=results_random_forest(data_rad_num,50,200,'frst_2_cor.p')
y_pred_val_t_3, rmse_val_t_3=results_random_forest(data_rad_num,50,500,'frst_3_cor.p')

y_pred_val_d_t_1, rmse_val_d_t_1=results_random_forest(data_rad_dum,50,500,'frst_d_1_cor.p')
y_pred_val_d_t_2, rmse_val_d_t_2=results_random_forest(data_rad_dum,50,300,'frst_d_2_cor.p')
y_pred_val_d_t_3, rmse_val_d_t_3=results_random_forest(data_rad_dum,50,200,'frst_d_3_cor.p')

y_pred_val_dw_t_1, rmse_val_dw_t_1=results_random_forest(data_rad_dumw,50,300,'frst_dw_1_cor.p')
y_pred_val_dw_t_2, rmse_val_dw_t_2=results_random_forest(data_rad_dumw,50,500,'frst_dw_2_cor.p')
y_pred_val_dw_t_3, rmse_val_dw_t_3=results_random_forest(data_rad_dumw,50,200,'frst_dw_3_cor.p')
#%%
y_pred_test, rmse_test=results_random_forest_test(data_rad_num,50,300,'frst_50,300.p')
#%%
y_pred_val_t_m, rmse_val_t_m=results_random_forest_minmax(data_rad_num,100,200)
y_pred_val_d_t_m_1, rmse_val_d_t_m_1=results_random_forest_minmax(data_rad_dum,50,200)
y_pred_val_dw_t_m, rmse_val_dw_t_m=results_random_forest_minmax(data_rad_dumw,50,50)
#%%
y_train_true=y=np.array(data_rad_num['Radiance'].loc['2017-01-24 13:00:00':'2018-12-31 18:00:00'])
y_min=min(y_train_true)
y_max=max(y_train_true)
error=scaler_y_t.inverse_transform(np.array(0.016).reshape(1,-1))
error=(error)*(y_max-y_min)
#%%
Rad_pred=rad_pred_val(y_val_t,y_pred_val_t_1)
Rad_pred.to_csv('Rad_pred_RF_50_300.csv', sep=";", decimal=",")
#%%
plt.figure()
Rad_pred.plot()
plt.show()
#%%
#frst_d_1=pickle.load(open('frst_d_1.p','rb'))
frst_1=RandomForestRegressor(min_samples_split=50,n_estimators=300).fit(X_train_d_t,y_train_d_t)
y_prevision_pred=frst_1.predict(X_prevision)
y_prevision_pred_hour=frst_1.predict(X_prevision_hour)
#Rad_pred_1h_3h=comp_rad(X_prevision,X_prevision_hour,frst_1)
#%%
dat_pred=pd.date_range('2019-04-06 05:00:00','2019-05-09 18:00:00',freq='H')
Date_pred=pd.DataFrame({'Date':dat_pred},index=dat_pred)
Date_pred['hour']=Date_pred.index.hour
Date_pred_rad=night_mask(Date_pred)
rad_pred_hour=pd.DataFrame(data={'Radiance_pred':y_prevision_pred_hour},index=Date_pred_rad.index)
Rad_pred_hour=rad_pred_hour.merge(Date_pred,how='right',left_index=True,right_index=True)
Rad_pred_hour=Rad_pred_hour.drop('Date',axis=1)
Rad_pred_hour=Rad_pred_hour.fillna(0)
Rad_pred_hour.to_csv('Radiance_prediction_owm_hourly.csv', sep=";", decimal=",")
#%%

y_true=data_rad_num['Radiance'].loc['2019-01-21 05:00:00':'2019-02-12 18:00:00']
y_true_hour=rad['Radiance'].loc['2019-04-06 05:00:00':'2019-05-09 18:00:00']
#RMSE_prevision=rmse(y_true,y_prevision_pred)
#Rad_prev=rad_pred_prev(y_true,y_prevision_pred)
#RMSE_prevision_hour=rmse(y_true_hour,y_prevision_pred_hour)
#Rad_prev_hour=rad_pred_prev(y_true_hour,y_prevision_pred_hour)