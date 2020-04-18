import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.svm import SVR

from utils import get_weights, write_weights
from utils import load_train_weather_data, load_train_load_data, load_day_type_data

def model_train_guangzhou():
    # 读取数据
    weights = get_weights('../data/pload_for_match/广州_统调负荷.csv')
    write_weights(weights, '../model/guangzhou/weights.csv')

    start_date = '09-28'
    end_date = '10-13'
    load_data = load_train_load_data('../data/pload_for_match/广州_统调负荷.csv', start_date, end_date)
    weather_data = load_train_weather_data('../data/weather_station/广州_气象站.csv', start_date, end_date)
    day_type_data = load_day_type_data('../holiday_info.csv', start_date, end_date)
    
    year_len = weather_data.shape[0]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    train_x = []
    for i in range(year_len):
        for j in range(weather_data[i].shape[0]):
            train_x.append([
                float(weather_data[i][j, 1]), 
                int(day_type_data[i][j, 1]), 
                int(day_type_data[i][j, 2]), 
                int(day_type_data[i][j, 3]), 
                int(weather_data[i][j, 0][:4]),
                int(weather_data[i][j, 0][5:7]),
                int(weather_data[i][j, 0][8:10]),
            ])
    train_x = np.array(train_x)
    scale_train_x = scaler_x.fit_transform(train_x)

    train_y = []
    for i in range(year_len):
        train_y.append(load_data[i][:, 1].astype(float))
    scale_train_y = scaler_y.fit_transform(np.array(train_y).reshape([-1, 1]))


    joblib.dump(scaler_x, '../model/guangzhou/scaler_x')
    joblib.dump(scaler_y, '../model/guangzhou/scaler_y')

    kernel = 'rbf'
    model = SVR(gamma='auto', kernel=kernel)
    model.fit(scale_train_x, scale_train_y.flatten())

    joblib.dump(model, '../model/guangzhou/train_model')
    
    
    
    
    
    