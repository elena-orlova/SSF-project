import xarray as xr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression


def model_performance(y_pred_test, y_test, num_loc=3274):
    mse_errors = np.mean((y_pred_test - y_test) ** 2, axis=0)
    output = [np.mean(mse_errors), np.std(mse_errors), 
              np.std(mse_errors)/ np.sqrt(num_loc),
              *np.quantile(mse_errors, [0.5, 0.75, 0.9])]
    return np.round(output, 2)


def avg_r2(precip_US_numpy_cropped_US, pred_all_cropped_US, precip_US_mask_cropped_US):
    N, M = precip_US_mask_cropped_US.shape[0], precip_US_mask_cropped_US.shape[1]
    unet_r2_all_train = np.zeros((N, M))
    
    if precip_US_numpy_cropped_US.shape[1] != N or pred_all_cropped_US.shape[2] != M:
        print('CHECK SHAPES')
    for i in range(N):
        for j in range(M):
            if precip_US_mask_cropped_US[i, j] == 1:
                unet_r2_all_train[i,j] = r2_score(precip_US_numpy_cropped_US[:,i,j],
                                                  pred_all_cropped_US[:,i,j])

    unet_r2_all_train[precip_US_mask_cropped_US == 0] = float("NaN")
    print('Avg r2 : {:.2f} '.format(unet_r2_all_train[precip_US_mask_cropped_US == 1].mean()))

    print('Median r2: {:.2f}  '.format(
          np.median(unet_r2_all_train[precip_US_mask_cropped_US == 1]), ))
    return unet_r2_all_train


def convert_1d_to_2d(images_vec_all, mask_cropped_US):
    images_2d = np.zeros((1,48,115))
    for i in range(1):
        image_vec = images_vec_all[i*3274:(i+1)*3274]
        index_1d = 0
        for lat in range(48):
            for lon in range(115):
                if mask_cropped_US[lat,lon]==1:
                    image_vec[index_1d]
                    images_2d[i,lat,lon] = image_vec[index_1d]
                    index_1d += 1
    return images_2d


def apply_mask_nan(data, mask):
    out = data
    out[:,mask==0] = np.nan
    return out


def create_classif_labels_tercile(data, pctl_33, pctl_66, mask):
    out = np.ones(data.shape)
    out[data<pctl_33] = 0
    out[data>pctl_66] = 2
    out = apply_mask_nan(out,mask)
    return out


def report_terc_scores(truth, preds, num_loc=3274):
    acc_scores = np.zeros((num_loc))
    for i in range(num_loc):
        acc_scores[i] = accuracy_score(truth[:, i], preds[:, i])
    mean_acc = acc_scores.mean()
    med_acc = np.median(acc_scores)
    se_acc = acc_scores.std() / np.sqrt(num_loc)
    print('Mean acc, median acc, SE: {:.2f} {:.2f} {:.2f}'.format(mean_acc*100, med_acc*100, se_acc*100))
#     acc_scores[mask==0] = np.float('NaN')
    return acc_scores


def get_LogRegr(X, y, mask, train_size=249, return_proba=False):
    N, M = mask.shape[0], mask.shape[1]
    w_preds = np.zeros((X.shape[0], N, M))
    w_probs = np.zeros((X.shape[0], 3, N, M))
    if return_proba:
        for i in range(N):
            for j in range(M):
                if mask[i, j] == 1:
                    reg = LogisticRegression(solver='liblinear').fit(X[:train_size, :, i, j], y[:train_size, i, j])
                    w_preds[:, i, j] = reg.predict(X[:,:,i, j])
                    w_probs[:, :, i, j] = reg.predict_proba(X[:,:,i, j])
        return w_preds, w_probs
    else:
        for i in range(N):
            for j in range(M):
                if mask[i, j] == 1:
                    reg = LogisticRegression(solver='liblinear').fit(X[:train_size, :, i, j], y[:train_size, i, j])
                    w_preds[:, i, j] = reg.predict(X[:,:,i, j])
        return w_preds


def sinusoid_positional_encoding_ref(length, dimensions, coord):
    def get_position_angle_vec(position):
        return [coord[position] / np.power(10000, 2 * (i // 2) / dimensions)
                for i in range(dimensions)]
 
    PE = np.array([get_position_angle_vec(i) for i in range(length)])
    PE[:, 0::2] = np.sin(PE[:, 0::2])  # dim 2i
    PE[:, 1::2] = np.cos(PE[:, 1::2])  # dim 2i+1
    return PE


def get_weighted_pred_LR_test(X, y, mask, train_size=249):
    N, M = mask.shape[0], mask.shape[1]
    w_preds = np.zeros((X.shape[0], N, M))
    
    for i in range(N):
        for j in range(M):
            if mask[i, j] == 1:
                reg = LinearRegression().fit(X[:train_size, :, i, j], y[:train_size, i, j])
                w_preds[:, i, j] = reg.predict(X[:,:,i,j])
    return w_preds


class generate_RF_predictions():
    def __init__(self, X_ftrs, precip_US_numpy, US_mask_numpy, start_date='1/1/1985', 
                 train_size = 249,
                lat_max=52.75, lat_min=21.25, lon_max=233.25, lon_min=296.75, 
                PE_dim=12, drop_positional_encodings=True, positional_encoding_channels=range(33,33+2*12),
                include_lat_lon_ftrs=False, standardize_forecasts_append_moments=False, drop_ens_members=False):
        
        # Convert mask from 0s and 1s to NaNs and 1s
        US_mask_nan = US_mask_numpy.copy()
        US_mask_nan[US_mask_nan==0] = np.nan
        
        # Define ranges of lat, lon, and date values
        lat_range = np.linspace(52.75, 21.25, num=X_ftrs.shape[2])
        lon_range = np.linspace(233.25, 296.75, num=X_ftrs.shape[3])
        date_range = pd.date_range(start_date, periods=X_ftrs.shape[0], freq='MS')
        
        # Turn our ftrs array and response array into xarray DataArrays
        X_ftrs_xr = xr.DataArray(X_ftrs, coords={"date":date_range,
                                          "channels": range(X_ftrs.shape[1]), "lat":lat_range,
                                          "lon":lon_range})\
                        .to_dataset(dim="channels")
        y_xr = xr.DataArray(precip_US_numpy, coords={"date":date_range,
                                            "lat":lat_range,
                                          "lon":lon_range})
        
        # make off-land values nan
        X_ftrs_xr = X_ftrs_xr * US_mask_nan
        y_xr = y_xr * US_mask_nan

        # drop positional encoding features
        if drop_positional_encodings:
            X_ftrs_xr = X_ftrs_xr.drop(positional_encoding_channels)
        
        # split up the DataArrays into training and test
        self.y_train_xr = y_xr.isel(date=slice(None, train_size))
        self.y_test_xr = y_xr.isel(date=slice(train_size, None))
        self.X_train_xr = X_ftrs_xr.isel(date=slice(None, train_size))
        self.X_test_xr = X_ftrs_xr.isel(date=slice(train_size, None))
        
        # convert DataArrays into Pandas DataFrames
        self.y_train_rf = self.y_train_xr.to_dataframe('precip').dropna()
        self.y_test_rf = self.y_test_xr.to_dataframe('precip').dropna()
        
        # if we want to include lat,lon as features, turn them from indices into values
        if include_lat_lon_ftrs:
            self.X_train_rf = self.X_train_xr.to_dataframe().dropna().reset_index(['lat','lon'])
            self.X_test_rf = self.X_test_xr.to_dataframe().dropna().reset_index(['lat','lon'])
        else:
            self.X_train_rf = self.X_train_xr.to_dataframe().dropna()
            self.X_test_rf = self.X_test_xr.to_dataframe().dropna()
            
        if standardize_forecasts_append_moments:
            self.X_train_rf['ens_mean'] = self.X_train_rf[range(24)].mean(axis=1)
            self.X_train_rf['ens_std'] = self.X_train_rf[range(24)].std(axis=1)
            self.X_train_rf.loc[:,range(24)] = self.X_train_rf[range(24)].sub(self.X_train_rf['ens_mean'], axis=0)\
                                                .div(self.X_train_rf['ens_std'], axis=0).fillna(0)
            
            self.X_test_rf['ens_mean'] = self.X_test_rf[range(24)].mean(axis=1)
            self.X_test_rf['ens_std'] = self.X_test_rf[range(24)].std(axis=1)
            self.X_test_rf.loc[:,range(24)] = self.X_test_rf[range(24)].sub(self.X_test_rf['ens_mean'], axis=0)\
                                                .div(self.X_test_rf['ens_std'], axis=0).fillna(0)
            
        
        if drop_ens_members:
            self.X_train_rf.drop(range(24),axis=1, inplace=True)
            self.X_test_rf.drop(range(24),axis=1, inplace=True)
        
    def return_X_dataframes(self, include='test'):
        # return the X_train and X_test dataframes to train an RF. For auditing purposes. 
        
        if include == 'test':
            return self.X_test_rf
        
        return self.X_train_rf, self.X_test_rf
    
    def return_y_dataframes(self, include='test'):
        # return the y_train and y_test dataframes to train an RF. For auditing purposes. 
        
        if include == 'test':
            return self.y_test_rf
        
        return self.y_train_rf, self.y_test_rf
        
    def train_model(self, rf=RandomForestRegressor(min_samples_leaf=1, max_features='sqrt', random_state=42, 
                                                   oob_score=True, n_estimators=100, n_jobs=-1)):
        # Core method. Trains a random forest on the data, given an untrained sklearn RF object
        rf.fit(self.X_train_rf, self.y_train_rf.values.ravel())
        
        self.rf = rf
    
    def return_predictions(self, include='test', output_format='grid', output_original_grid=False):
        # Core method. After using the train_model method, returns predictions, either as a vector or a grid
        
        
        if include == 'test':
            y_preds_test = self.y_test_rf.copy()
            y_preds_test['precip'] = self.rf.predict(self.X_test_rf)
            
            if output_format != 'grid':
                return y_preds_test
            
            if output_original_grid:
                grid_preds_test = xr.Dataset.from_dataframe(y_preds_test).sortby(['lon'], ascending=True)['precip']\
                                            .interp_like(self.y_test_xr, method='nearest').to_numpy()
                return grid_preds_test
            
            grid_preds_test = xr.Dataset.from_dataframe(y_preds_test).sortby(['lon'], ascending=True)['precip'].to_numpy()
            return grid_preds_test
        
        if include == 'all':
            y_preds_test = self.y_test_rf.copy()
            y_preds_test['precip'] = self.rf.predict(self.X_test_rf)
            y_preds_train = self.y_train_rf.copy()
            y_preds_train['precip'] = self.rf.predict(self.X_train_rf)
            
            if output_format != 'grid':
                return y_preds_train, y_preds_test
            
            if output_original_grid:
                grid_preds_test = xr.Dataset.from_dataframe(y_preds_test).sortby(['lon'], ascending=True)['precip']\
                                            .interp_like(self.y_test_xr, method='nearest').to_numpy()
                grid_preds_train = xr.Dataset.from_dataframe(y_preds_train).sortby(['lon'], ascending=True)['precip']\
                                            .interp_like(self.y_train_xr, method='nearest').to_numpy()
                return np.concatenate((grid_preds_train, grid_preds_test))
            
            grid_preds_test = xr.Dataset.from_dataframe(y_preds_test).sortby(['lon'], ascending=True)['precip'].to_numpy()
            grid_preds_train = xr.Dataset.from_dataframe(y_preds_train).sortby(['lon'], ascending=True)['precip'].to_numpy()
            return np.concatenate((grid_preds_train, grid_preds_test))