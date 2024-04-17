import pandas as pd
import numpy as np

import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning as L

from sklearn.preprocessing import MinMaxScaler

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import joblib

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error


target_col = 0

class WeatherDataModule(L.LightningDataModule):
    def __init__(self, data_dir="data\current_weather_data.csv", index_='timestamp', 
                 column=0, batch_size=64, window_size=5, normalize_=False,
                 date_range = None, step_ = 24, return_tensor = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.index_ = index_
        self.column = column
        self.date_range = date_range
        self.window_size = window_size
        self.step_ = step_
        self.return_tensor = return_tensor

        self.normalize_ = normalize_

    def prepare_data(self):
        df_ = pd.read_csv(self.data_dir, index_col=self.index_, parse_dates=True)
        if self.date_range != None:
            df_ = df_[self.date_range]
        
        if self.column == None:
            self.df = df_
        else:
            self.df = df_.iloc[:,self.column]
        
        if self.normalize_:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.df.values.reshape(-1, 1))
            self.df = self.normalize(self.df)

        self.windows, self.targets = self.window_step(self.df, self.step_)
        self.windows, self.targets = self.windows.squeeze(), self.targets.squeeze()

    def window_step(self, dataset, step_ ):
        """Transform a time series into a prediction dataset
        
        Args:
            dataset: A numpy array of time series, first dimension is the time steps
            lookback: Size of window for prediction
        """
        X, y = [], []
        for i in range(len(dataset)-self.window_size - step_):
            feature = dataset[i:i+self.window_size]
            target = dataset[i+self.window_size:i+self.window_size+step_]
            X.append(feature)
            y.append(target)

        X_r = torch.tensor(np.array(X))
        y_r = torch.tensor(np.array(y))

        if self.return_tensor:
            return X_r.float(), y_r.float()
        else:
            return np.array(X), np.array(y)
    
    def normalize(self, series):
        if self.column == None:
            return pd.DataFrame(self.scaler.fit_transform(series), index=series.index)
        else:
            return pd.DataFrame(self.scaler.fit_transform(series.values.reshape(-1, 1)), index=series.index)
    
    def inverse_normalze(self, series):
        if self.column == None:
            return pd.DataFrame(self.scaler.inverse_transform(series), index=series.index)
        else:
            return pd.DataFrame(self.scaler.inverse_transform(series.values.reshape(-1, 1)), index=series.index)
    
    def inverse_single_column(self, series):
        if self.column == None:
            zeros_ = pd.DataFrame(np.zeros((series.shape[0], self.df.shape[1])))
            zeros_[target_col] = series 
            return pd.DataFrame(self.scaler.inverse_transform(zeros_))[target_col]
        else:
            return pd.DataFrame(self.scaler.inverse_transform(series.reshape(-1, 1)))

    def setup(self, stage: str):
        self.split = [round(len(self.df) * 0.7), round(len(self.df) * 0.9)]

        self.f_train, self.t_train = self.windows[:self.split[0]], self.targets[:self.split[0]]
        self.f_valid, self.t_valid = self.windows[self.split[0]:self.split[1]], self.targets[self.split[0]:self.split[1]]
        self.f_test, self.t_test = self.windows[self.split[1]:], self.targets[self.split[1]:]

        print(f'Train: {self.f_train.shape}\nValid: {self.f_valid.shape}\nTest: {self.f_test.shape}')

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.f_train, self.t_train), batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.f_train, self.t_train), batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.f_train, self.t_train), batch_size=self.batch_size, shuffle=False)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size, shuffle=False)

def get_dates(weeks_ = 52):
    today = datetime.now()

    today_rounded_down = today.replace(minute=0, second=0, microsecond=0)

    one_year_ago = today - timedelta(days=weeks_ * 7 + 1)

    one_year_ago_rounded = one_year_ago.replace(minute=0, second=0, microsecond=0)

    if one_year_ago.minute != 0 or one_year_ago.second != 0 or one_year_ago.microsecond != 0:
        one_year_ago_rounded += timedelta(hours=1)

    formatted_today_rounded_down = today_rounded_down.strftime("%Y-%m-%dT%H:%M:%S")
    formatted_one_year_ago_rounded = one_year_ago_rounded.strftime("%Y-%m-%dT%H:%M:%S")

    return formatted_one_year_ago_rounded, formatted_today_rounded_down, today

def train(dm= None, folder='models_2', train_models=True, rfr=True, xgb_=True, knn=True, ridge=True, window_size=24, step=1):
    if dm.column == None:
        uni_multi = 'multi'
    else:
        uni_multi = 'uni'

    if train_models:

        X_ = dm.f_train.reshape(-1, window_size * dm.df.shape[1])
        X_valid = dm.f_valid.reshape(-1, window_size * dm.df.shape[1])

        if dm.column == None:
            y_ = dm.t_train[:,:,target_col]
            y_valid = dm.t_valid[:,:,target_col]
        else:
            y_ = dm.t_train
            y_valid = dm.t_valid


        if rfr:
            start_time = time.time()

            print('Training Random Forest Regressor...')

            rf_regressor_1 = RandomForestRegressor(n_estimators=50, random_state=42)       
        
                
            rf_regressor_1.fit(X_, y_)

            end_time = time.time()

            elapsed_minutes = (end_time - start_time) / 60
            print(f"Elapsed minutes: {elapsed_minutes}")
            print('\n\n')

            joblib.dump(rf_regressor_1, f'{folder}/random_forest_model_{uni_multi}_ws_{window_size}_s_{step}.pkl')
                
        if xgb_:
            start_time = time.time()

            print('Training XGBoost Model...')
            
            rf_regressor_xgb = xgb.XGBRegressor(base_score=0.5, booster='gbtree', learning_rate=0.01,
                                                    max_depth=3, n_estimators=1000,
                                                    objective='reg:linear', random_state=0)
        
        

            rf_regressor_xgb.fit(X_, y_, eval_set=[(X_valid, y_valid)], 
                                early_stopping_rounds=10, 
                                verbose=False)

            end_time = time.time()

            elapsed_minutes = (end_time - start_time) / 60
            print(f"Elapsed minutes: {elapsed_minutes}")
            print('\n\n')

            model_path = f"{folder}/xgboost_model_{uni_multi}_ws_{window_size}_s_{step}.bin"
            rf_regressor_xgb.save_model(model_path)  
                
        if knn:
            start_time = time.time()
            
            print('Training KNN Regressor...')
            knn_regressor = KNeighborsRegressor(n_neighbors=10) 
            knn_regressor.fit(X_, y_)

            end_time = time.time()

            elapsed_minutes = (end_time - start_time) / 60
            print(f"Elapsed minutes: {elapsed_minutes}")
            print('\n\n')

            joblib.dump(knn_regressor, f'{folder}/knn_regressor_model_{uni_multi}_ws_{window_size}_s_{step}.pkl')

        if ridge:
            start_time = time.time()

            print('Training ridge Regressor...')
            
            ridge_model = Ridge(alpha=0.15)  

            ridge_model.fit(X_, y_)

            end_time = time.time()

            elapsed_minutes = (end_time - start_time) / 60 
            print(f"Elapsed minutes: {elapsed_minutes}")    
            print('\n\n')

            joblib.dump(ridge_model, f'{folder}/ridge_regressor_model_{uni_multi}_ws_{window_size}_s_{step}.pkl')

def load_models(dm = None, folder='models_2', window_size=24, step=1):
    if dm.column == None:
        uni_multi = 'multi'
    else:
        uni_multi = 'uni'

    rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr_model = joblib.load(f'{folder}/random_forest_model_{uni_multi}_ws_{window_size}_s_{step}.pkl')

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f'{folder}/xgboost_model_{uni_multi}_ws_{window_size}_s_{step}.bin')
    
    knn_model = joblib.load(f'{folder}/knn_regressor_model_{uni_multi}_ws_{window_size}_s_{step}.pkl')

    ridge_model = joblib.load(f'{folder}/ridge_regressor_model_{uni_multi}_ws_{window_size}_s_{step}.pkl')

    print('Models loaded...')

    return rfr_model, xgb_model, knn_model, ridge_model

def plot_results(seed, height, width, interval, X, y, rfr_model, xgb_model, knn_model, ridge_model,
                  plot_features=True, metrics=True, weights_ = [0.25, 0.25, 0.25, 0.25], window_size=24, step=1, dm=None):

    mse_rfr = []
    mse_xgb = []
    mse_knn = []
    mse_ridge = []
    mse_avg = []

    mse_tracker = {'Random Forest': mse_rfr, 'XGBoost': mse_xgb, 'kNN': mse_knn, 'Ridge': mse_ridge, 'Average': mse_avg}

    fig, axes = plt.subplots(height,width, figsize=(18, 3 * height))

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            seed_index = i * 2 + j
            seed = seed_index * interval 
            current_data = X[seed]

            current_data_ = current_data.reshape(1, window_size * dm.df.shape[1])

            step_pred_rfr = rfr_model.predict(current_data_).squeeze()
            step_pred_xgb = xgb_model.predict(current_data_).squeeze()
            step_pred_knn = knn_model.predict(current_data_).squeeze()
            step_pred_ridge = ridge_model.predict(current_data_).squeeze()

            average = step_pred_rfr * weights_[0] + step_pred_xgb * weights_[1] + step_pred_knn * weights_[2] + step_pred_ridge * weights_[3]

            if dm.column == None:
                t_test_data = y[seed:seed + step][0][:, target_col]
                current_data = current_data[:, target_col]
            else:
                t_test_data = y[seed]

            if plot_features:
                ax.plot(range(window_size), current_data[:, 0], label='temperature', c='gray')
                ax.plot(range(window_size), current_data[:, 1], label='humidity', c='gray')
                ax.plot(range(window_size), current_data[:, 3], label='wind_direction', c='gray')
                ax.plot(range(window_size), current_data[:, 4], label='wind_gusts', c='gray')

            if dm.normalize_: 
                ax.plot(range(window_size), dm.inverse_single_column(current_data), label='wind_speed', c='black')
                ax.plot(range(window_size, window_size + step), dm.inverse_single_column(t_test_data), label = 'Target', c='blue')
                ax.plot(range(window_size, window_size + step), dm.inverse_single_column(step_pred_rfr), label = 'Reg', c='green')
                ax.plot(range(window_size, window_size + step), dm.inverse_single_column(step_pred_xgb), label = 'XGB', c='red')
                ax.plot(range(window_size, window_size + step), dm.inverse_single_column(step_pred_knn), label = 'kNN', c='violet')
                ax.plot(range(window_size, window_size + step), dm.inverse_single_column(step_pred_ridge), label = 'Ridge', c='yellow')
                ax.plot(range(window_size, window_size + step), dm.inverse_single_column(average), label = 'Average', c='orange', linewidth=2)

                mse_rfr.append(mean_squared_error(dm.inverse_single_column(t_test_data), dm.inverse_single_column(step_pred_rfr)))
                mse_xgb.append(mean_squared_error(dm.inverse_single_column(t_test_data), dm.inverse_single_column(step_pred_xgb)))
                mse_knn.append(mean_squared_error(dm.inverse_single_column(t_test_data), dm.inverse_single_column(step_pred_knn)))
                mse_ridge.append(mean_squared_error(dm.inverse_single_column(t_test_data), dm.inverse_single_column(step_pred_ridge)))
                mse_avg.append(mean_squared_error(dm.inverse_single_column(t_test_data), dm.inverse_single_column(average)))

            else:
                ax.plot(range(window_size), current_data, label='wind_speed', c='black')
                ax.plot(range(window_size, window_size + step), t_test_data, label = 'Target', c='blue')
                ax.plot(range(window_size, window_size + step), step_pred_rfr, label = 'Reg', c='green')
                ax.plot(range(window_size, window_size + step), step_pred_xgb, label = 'XGB', c='red')     
                ax.plot(range(window_size, window_size + step), step_pred_knn, label = 'kNN', c='violet')
                ax.plot(range(window_size, window_size + step), step_pred_ridge, label = 'Ridge', c='yellow')
                ax.plot(range(window_size, window_size + step), average, label = 'Average', c='orange', linewidth=2)

                mse_rfr.append(mean_squared_error(t_test_data, step_pred_rfr))
                mse_xgb.append(mean_squared_error(t_test_data, step_pred_xgb))
                mse_knn.append(mean_squared_error(t_test_data, step_pred_knn))
                mse_ridge.append(mean_squared_error(t_test_data, step_pred_ridge))
                mse_avg.append(mean_squared_error(t_test_data, average))

            if i == 0 and j == 0:  
                ax.legend(loc='upper left')

            ax.set_title(f"Seed: {seed}")

    if metrics:
        for key, value in mse_tracker.items():
            print(f'Mean MSE for {key}: {np.mean(value)}')
        

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(mse_tracker.keys(), [np.mean(value) for value in mse_tracker.values()])
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE for Different Models')
    plt.show()

    return [np.mean(value) for value in mse_tracker.values()]

def metrics(column_, X, y, rfr_model, xgb_model, knn_model, ridge_model, window_size, col_):    
    weights = [0.25, 0.25, 0.25, 0.25]

    if column_ == None:
        y_true = y[:, :, target_col]
    else:
        y_true = y

    X_valid = X.reshape(-1, window_size * col_)

    y_pred_knn = knn_model.predict(X_valid)
    y_pred_ridge = ridge_model.predict(X_valid)
    y_pred_rfr = rfr_model.predict(X_valid)
    y_pred_xgb = xgb_model.predict(X_valid)
    y_pred_avg = y_pred_rfr * weights[0] + y_pred_xgb * weights[1] + y_pred_knn * weights[2] + y_pred_ridge * weights[3]


    mse_avg = mean_squared_error(y_true, y_pred_avg)
    mse_knn = mean_squared_error(y_true, y_pred_knn)
    mse_ridge = mean_squared_error(y_true, y_pred_ridge)
    mse_rfr = mean_squared_error(y_true, y_pred_rfr)
    mse_xgb = mean_squared_error(y_true, y_pred_xgb)

    # Print MSE for each model
    print("MSE for Average model:", mse_avg)
    print("MSE for kNN model:", mse_knn)
    print("MSE for Ridge model:", mse_ridge)
    print("MSE for Random Forest model:", mse_rfr)
    print("MSE for XGBoost model:", mse_xgb)

    plt.figure(figsize=(8, 6))  
    plt.bar(['Average', 'kNN', 'Ridge', 'Random Forest', 'XGBoost'], [mse_avg, mse_knn, mse_ridge, mse_rfr, mse_xgb])
    plt.xlabel('Models')
    plt.ylabel('Normalized Mean Squared Error (MSE)')
    plt.title('MSE for Different Models')
    plt.show()

def build_model(hidden_size = [64, 32], out = 1, input_shape_ = 24 * 2, type_ = 'DNN'):
    if type_ == 'DNN':
        model = Sequential()
        model.add(Dense(hidden_size[0], input_shape=(input_shape_,)))
        model.add(Dense(hidden_size[1], activation='relu'))
        model.add(Dense(out))
    
    elif type_ == 'LSTM':
        model = Sequential()
        model.add(LSTM(hidden_size[0], input_shape=(input_shape_, 1)))
        # model.add(Dense(hidden_size[1], activation='relu'))
        model.add(Dense(out))
    
    elif type_ == 'GRU':
        model = Sequential()
        model.add(GRU(hidden_size[0], input_shape=(input_shape_, 1)))
        # model.add(Dense(hidden_size[1], activation='relu'))
        model.add(Dense(out))

    elif type_ == 'CNN':
        '''
        The CNN is more effective when using less layers and filters

        Hyperparameters will include:
        - Number of filters or hidden size
        - Kernel size
        - Total layers
        '''
        model = tf.keras.Sequential([
            Conv1D(filters=hidden_size[0], kernel_size=3, activation='relu', input_shape=(input_shape_, 1)),

            Conv1D(filters=hidden_size[0], kernel_size=3, activation='relu'),

            MaxPooling1D(pool_size=2),

            Conv1D(filters=hidden_size[1], kernel_size=3, activation='relu'),

            Conv1D(filters=hidden_size[1], kernel_size=3, activation='relu'),

            MaxPooling1D(pool_size=2),

            Flatten(),

            Dense(512, activation='relu'),

            Dropout(0.5),

            Dense(out, activation='sigmoid')
        ])

    return model

