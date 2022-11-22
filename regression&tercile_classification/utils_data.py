import numpy as np
import pandas as pd


def get_train_val_ens_target(target='precip_ncep', path='/share/data/willett-group/climate/'):
    
    if target == 'precip_ncep':
        ens_forecasts_all = np.load(path + 'train_val/NCEP_precip_ens.npy') 
        target_rect = np.load(path + 'train_val/precip_US.npy')
        target_for_lags = np.load(path + 'train_val/precip_US_1983_2010.npy') 
        
    elif target == 'tmp_ncep': 
        # here everything is in Celcius
        ens_forecasts_all = np.load(path + 'train_val/NCEP_tmp_ens.npy') 
        target_rect = np.load(path + 'train_val/tmp_US_1983_2010.npy')
        target_rect = target_rect[24:] # start from 1985
        target_for_lags = np.load(path + 'train_val/tmp_US_1983_2010.npy')
        
    elif target == 'precip_nasa': 
        ens_forecasts_all = np.load(path + 'train_val/NASA_precip_ens.npy')
        ens_forecasts_all = ens_forecasts_all[:, :-1]
        target_rect = np.load(path + 'train_val/precip_US.npy') 
        target_for_lags = np.load(path + 'train_val/precip_US_1983_2010.npy')
        
    elif target == 'tmp_nasa': 
        ens_forecasts_all = np.load(path + 'train_val/NASA_tmp_ens.npy') 
        ens_forecasts_all = ens_forecasts_all[:, :-1]
        ens_forecasts_all = ens_forecasts_all - 273.15 # convert to Celcius
        target_rect = np.load(path + 'train_val/tmp_US_1983_2010.npy') 
        target_rect = target_rect[24:] # start from 1985
        target_for_lags = np.load(path + 'train_val/tmp_US_1983_2010.npy')
        
    else:
        print('NO DATA (check target value)')
        
    return ens_forecasts_all, target_rect, target_for_lags


def get_train_val_obs_data(target='precip_ncep', path='/share/data/willett-group/climate/'):
    
    if target == 'precip_ncep' or target == 'precip_nasa':
        tmp_ssts = np.load(path + 'train_val/tmp_US_1984_2010.npy')
        
    else:
        tmp_ssts = np.load(path + 'train_val/precip_US_1984_2010.npy') 
        
    rhum = np.load(path + 'train_val/rhum_US_1984_2010.npy') 
    ght = np.load(path + 'train_val/hgt500_1984_2010.npy') 
    pressure = np.load(path + 'train_val/slp_1984_2010.npy') 
    
    sst_train = np.array(pd.read_pickle(path + 'train_val/sst_train_2mo_offset.pkl')) 
    sst_val = np.array(pd.read_pickle(path + 'train_val/sst_val_2mo_offset.pkl')) 
    
    return tmp_ssts, rhum, ght, pressure, sst_train, sst_val


def get_obs_trend(target='precip_ncep', path='data/'):
    
    if target == 'precip_ncep' or target == 'precip_nasa':
        print('precip')
        trend = np.load(path + 'train_val/long_term_avg_precip.npy') 
        
    elif target == 'tmp_ncep' or target == 'tmp_nasa':
        print('tmp')
        trend = np.load(path + 'train_val/long_term_avg_tmp.npy') 
        trend = trend - 273.15
    
    return trend


def get_test_ens_target(target='precip_ncep', path='/share/data/willett-group/climate/'):
    
    if target == 'precip_ncep':
        # ens members start from April 2011 till Dec 2020
        ens_forecasts_all_test = np.load(path + 'test/NCEP_precip_ens_test.npy') 
        target_test = np.load(path + 'test/precip_US_test.npy') 
        precip_US_rect_old = np.load(path + 'train_val/precip_US.npy') # 1985-2010 
        
    elif target == 'tmp_ncep': 
        ens_forecasts_all_test = np.load(path + 'test/NCEP_tmp_ens_test.npy') 
        target_test = np.load(path + 'test/tmp_US_test.npy') 
        precip_US_rect_old = np.load(path + 'train_val/tmp_US_1983_2010.npy') 
        
    elif target == 'precip_nasa': 
        # ens members start from Jan 2011 to Jan 2018
        ens_forecasts_all_test = np.load(path + 'test/NASA_precip_ens_test.npy')
        ens_forecasts_all_test = ens_forecasts_all_test[:, :-1]
        target_test = np.load(path + 'test/precip_US_2011_18.npy')  
        precip_US_rect_old = np.load(path + 'train_val/precip_US.npy') # 1985-2010 
        
    elif target == 'tmp_nasa': 
        ens_forecasts_all_test = np.load(path + 'test/NASA_tmp_ens_test.npy') 
        ens_forecasts_all_test = ens_forecasts_all_test[:, :-1]
        target_test = np.load(path + 'test/tmp_US_2011_18.npy') 
        precip_US_rect_old = np.load(path + 'train_val/tmp_US_1983_2010.npy') 
    else:
        print('NO DATA')
        
    return ens_forecasts_all_test, target_test, precip_US_rect_old


def get_test_obs_data(target='precip_ncep', path='/share/data/willett-group/climate/'):
    if target == 'precip_ncep':
        # available from Jan 2011 
        print('getting tmp')
        tmp_ssts_test = np.load(path + 'test/tmp_US_test.npy') 
    elif target == 'tmp_ncep':
        print('getting precip')
        tmp_ssts_test = np.load(path + 'test/precip_US_test.npy') 

    if target == 'precip_nasa':
        # available from Jan 2011 
        tmp_ssts_test = np.load(path + 'test/tmp_US_2011_18.npy')  
    elif target == 'tmp_nasa':
        tmp_ssts_test = np.load(path + 'test/precip_US_2011_18.npy')  

    # available from Jan 2011
    rhum_test = np.load(path + 'test/rhum_test_2011_2020.npy') 
    ght_test = np.load(path +'test/hgt500_test_2011_2020.npy') 
    pressure_test = np.load(path + 'test/slp_test_2011_2020.npy') 
    
    return tmp_ssts_test, rhum_test, ght_test, pressure_test


def get_quantiles_33_66(target='precip_ncep', path='/share/data/willett-group/climate/'):
    
    if target == 'precip_ncep' or target == 'precip_nasa':
        print('geting precip prctl')
        long_term_pctl_33 = np.load(path + 'train_val/long_term_pctl_33_precip.npy')
        long_term_pctl_66 = np.load(path + 'train_val/long_term_pctl_66_precip.npy') 
        
    else: 
        print('geting tmp prctl')
        long_term_pctl_33 = np.load(path + 'train_val/long_term_pctl_33_tmp.npy') 
        long_term_pctl_66 = np.load(path + 'train_val/long_term_pctl_66_tmp.npy') 
        
    return long_term_pctl_33, long_term_pctl_66


def get_ens_climatology(target='precip_ncep', path='/share/data/willett-group/climate/'):
    if target =='tmp_ncep':
        ens_climatology = np.load(path + 'train_val/ens_climatology_ncep_tmp.npy') 
    elif target =='precip_ncep':
        ens_climatology = np.load(path + 'train_val/ens_climatology_ncep_precip.npy') 
    elif target =='tmp_nasa':
        ens_climatology = np.load(path + 'train_val/ens_climatology_nasa_tmp.npy')  
    elif target =='precip_nasa':
        ens_climatology = np.load(path + 'train_val/ens_climatology_nasa_precip.npy') 
    else:
        print('NO DATA')
        
    return ens_climatology
