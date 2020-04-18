import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.svm import SVR

from utils import load_predict_load_data, load_predict_weather_data, load_weights, date_to_index
from utils import write_result, download_oss_data, upload_oss_result


def model_predict_guangzhou(date):
    scaler_x = joblib.load('../model/guangzhou/scaler_x')
    scaler_y = joblib.load('../model/guangzhou/scaler_y')
    model = joblib.load('../model/guangzhou/train_model')
    
    if download_oss_data('pload_for_match', '../data'):
        print('Pload download success.')
    else:
        print('Error! Pload download fail.')
    if download_oss_data('WeatherFiles', '../data'):
        print('WeatherFiles download success.')
    else:
        print('Error! WeatherFiles download fail.')

    weather_data = load_train_weather_data('../data/WeatherFiles/weather_forecast_hefeng/2019-%s_广州_hefeng.csv' % date, '2019-%s' % date)
    day_type_data = load_day_type_data('../holiday_info.csv', date, '10-13')

    predict_x = [weather_data,
                 int(day_type_data[-1][j, 1]), 
                 int(day_type_data[-1][j, 2]),
                 int(day_type_data[-1][j, 3]),
                 2019,
                 int(data[:2]),
                 int(data[3:]),
                ]
    predict_x = np.array(predict_x).reshape([-1, 1])
    scale_predict_x = scaler_x.transform(predict_x)
    
    scale_predict_y = model.predict(scale_predict_x).reshape([-1, 1])
    predict_y = scaler_y.inverse_transform(scale_predict_y).flatten()
    
    weights = load_weights('../model/guangzhou/weights.csv')
    result = weights[date_to_index(date)] * predict_y
    
    write_result(result, '2019-%s 00:00'%s, '广州', 'AI_31', file_path='../load_predict_day.csv')