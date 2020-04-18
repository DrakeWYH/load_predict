import datetime
import csv
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from pso import pso
from acquire_ossdata import AcquireOSS


def download_oss_data(folder_name, save_path):
    try:
        oss_download = AcquireOSS()
        bucket = oss_download.create_bucket()
        file_path_list = oss_download.get_file_path(bucket, folder_name)
        oss_download.file_download(bucket, file_path_list, save_path)
        return True
    except:
        return False
    

def upload_oss_result(upload_name, upload_path):
    try:
        oss_upload = AcquireOSS(bucket_name='ai-31')
        bucket = oss_upload.create_bucket()
        oss_upload.file_upload(bucket, upload_name, upload_path)
        return True
    except:
        return False


def quick_plot(test_y, predict_y, xticks, title='', save_path=''):
    plt.figure(figsize=(10, 3))
    plt.plot(test_y, 'k-', label='test')
    plt.plot(predict_y, 'r-', label='predict')
    plt.xticks(np.arange(predict_y.shape[0]), xticks)
    plt.legend()
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def date_format_change(date, old_format, new_format):
    date_tmp = datetime.datetime.strptime(date, old_format)
    date_new = date_tmp.strftime(new_format)
    return date_new


def datetime_change(start_time, move):
    timeStamp = int(time.mktime(time.strptime(start_time, '%Y-%m-%d %H:%M')))
    timeStamp += move * 15 * 60
    target_time = time.localtime(timeStamp)
    target_time = time.strftime('%Y-%m-%d %H:%M', target_time)
    return target_time


def datetime_delta(start_time, end_time):
    timeStamp_start = int(time.mktime(time.strptime(start_time, '%Y-%m-%d %H:%M')))
    timeStamp_end = int(time.mktime(time.strptime(end_time, '%Y-%m-%d %H:%M')))
    delta_time = int((timeStamp_end - timeStamp_start) / 15 / 60)
    return delta_time


def get_day_of_week(date, date_format):
    date_object = datetime.datetime.strptime(date, date_format)
    year = date_object.year
    month = date_object.month
    day = date_object.day
    return datetime.datetime(year, month, day).strftime('%w')


def load_train_load_data(file_name, start_date, end_date):
    with open(file_name, 'r', encoding='utf-8') as rf:
        csv_reader = csv.reader(rf)
        is_begin = False
        data = []
        tmp_year = ''
        data_year = []
        data_day = []
        for line in csv_reader:
            if line[0][:4] == '2008' or line[0][:4] == '2019':
                continue
            if line[0][:4] != tmp_year:
                if len(data_year):
                    data.append(np.array(data_year))
                    data_year = []
                tmp_year = line[0][:4]
            if not is_begin and line[0][5:10] == start_date:
                is_begin = True
            elif is_begin and line[0][5:10] == end_date:
                is_begin = False
            if is_begin:
                if line[0][11:16] == '00:00':
                    data_day = []
                data_day.append(line[1])
                if line[0][11:16] == '23:45':
                    data_year.append([line[0][:10], np.mean(np.array(data_day, dtype=np.float))])
        if len(data_year):
            data.append(np.array(data_year))
        return np.array(data)


def load_train_weather_data(file_name, start_date, end_date):
    with open(file_name, 'r', encoding='utf-8') as rf:
        csv_reader = csv.reader(rf)
        is_begin = False
        data_tmp = []
        for line in csv_reader:
            tmp = line[0].strip().replace('"', '').split(';')[:2]
            if len(tmp[0]) == 16 and tmp[0][:4] != '2019':
                tmp[0] = date_format_change(tmp[0], '%d.%m.%Y %H:%M', '%Y-%m-%d %H:%M')
                data_tmp.append(tmp)
    data_tmp = np.array(data_tmp)
    
    data = []
    tmp_year = ''
    data_year = []
    pre_date = ''
    data_day = []
    is_begin = False
    for i in range(data_tmp.shape[0] - 1, -1, -1):
        if data_tmp[i, 0][:4] != tmp_year:
            if len(data_day):
                data_year.append([pre_date, np.mean(np.array(data_day, dtype=np.float))])
                data_day = []
                pre_date = data_tmp[i, 0][5:10]
            if len(data_year):
                data.append(np.array(data_year))
                data_year = []
            tmp_year = data_tmp[i, 0][:4]
            
        if not is_begin and data_tmp[i, 0][5:10] == start_date:
            is_begin = True
        elif is_begin and data_tmp[i, 0][5:10] == end_date:
            is_begin = False
            if len(data_day):
                data_year.append([pre_date, np.mean(np.array(data_day, dtype=np.float))])
                data_day = []
            pre_date = data_tmp[i, 0][:10]
            
        if is_begin:
            if data_tmp[i, 0][:10] != pre_date:
                if len(data_day):
                    data_year.append([pre_date, np.mean(np.array(data_day, dtype=np.float))])
                    data_day = []
                pre_date = data_tmp[i, 0][:10]
            if data_tmp[i, 1]:
                data_day.append(data_tmp[i, 1])
    if len(data_day):
        data_year.append([pre_date, np.mean(np.array(data_day, dtype=np.float))])
    if len(data_year):
        data.append(np.array(data_year))
    return np.array(data)


def load_day_type_data(file_name, start_date, end_date):
    with open(file_name, 'r', encoding='utf-8') as rf:
        csv_reader = csv.reader(rf)
        next(csv_reader)
        data = []
        tmp_year = ''
        data_year = []
        is_begin = False
        for line in csv_reader:
            if line[0][:4] != tmp_year:
                if len(data_year):
                    data.append(np.array(data_year))
                    data_year = []
                tmp_year = line[0][:4]
            if not is_begin and line[0][5:10] == start_date:
                is_begin = True
            elif is_begin and line[0][5:10] == end_date:
                is_begin = False
            if is_begin and line:
                data_year.append(line)
        if len(data_year):
            data.append(np.array(data_year))
    return np.array(data)


def load_load_data(file_name, start_date, end_date):
    with open(file_name, 'r', encoding='utf-8') as rf:
        csv_reader = csv.reader(rf)
        is_begin = False
        data = []
        tmp_year = ''
        data_year = []
        for line in csv_reader:
            if line[0][:4] == '2019' or line[0][:4] == '2008':
                break
            if line[0][:4] != tmp_year:
                if len(data_year):
                    data.append(np.array(data_year))
                    data_year = []
                tmp_year = line[0][:4]
            if not is_begin and line[0][5:10] == start_date:
                is_begin = True
            elif is_begin and line[0][5:10] == end_date:
                is_begin = False
            if is_begin and line:
                data_year.append(np.array(line))
        if len(data_year):
            data.append(np.array(data_year))
        return np.array(data)
    
    
def load_predict_load_data(file_name, date):
    '''
    data = load_train_load_data('../data_load/广州/广州_统调负荷.csv', '2018-09-01')
    '''
    with open(file_name, 'r', encoding='utf-8') as rf:
        csv_reader = csv.reader(rf)
        data = []
        for line in csv_reader:
            if len(data) and line[0] == data[-1][0]:
                continue
            if line[0][:10] == date:
                data.append(line[1])
        if len(data) != 96:
            return None
        return np.mean(np.array(data, dtype=float))
    

def load_predict_weather_data(file_name, date):
    '''
    data = load_train_weather_data('../data_guangzhou/2019-08-07_广州_hefeng.csv', '2019-08-07')
    '''
    with open(file_name, 'r', encoding='utf-8') as rf:
        csv_reader = csv.reader(rf)
        next(csv_reader)
        data = []
        for line in csv_reader:
            if len(data) and line[0] == data[-1][0]:
                continue
            if line[0][:10] == date:
                data.append(line[8])
        if len(data) != 24:
            print(len(data))
            return None
    return np.mean(np.array(data, dtype=float))


def mycon(aorg_load, x):
    aorg_load_mean = aorg_load.mean()
    delta = aorg_load_mean * x - aorg_load
    ad = np.mean(np.abs(delta))
    return ad


def get_weights(file_name):
    load_data = load_load_data(file_name, '09-29', '10-13')
    load_data_value = load_data.reshape([-1, 96, 2])[:, :, 1].astype(float)
    aload = (load_data_value / np.tile(np.mean(load_data_value, axis=1, keepdims=True), (1,96)))

    lb = [0.5] * 96
    ub = [1.8] * 96
    weights = []
    for sample in range(154):
        xopt, fopt = pso(mycon, lb, ub, args=(aload[sample],),swarmsize=100, maxiter=50,omega=0.8,phip=0.8,phig=0.8)
        weights.append(xopt)
    weights = np.array(weights)
    weights = weights.reshape([-1, 14, 96])
    w1 = np.mean(weights, axis=0)
    return w1


def write_weights(weights, file_name):
    with open(file_name, 'w', encoding='utf-8', newline='') as wf:
        csv_writer = csv.writer(wf)
        for i in range(weights.shape[0]):
            csv_writer.writerow(weights[i])

            
def load_weights(file_name):
    with open(file_name, 'r', encoding='utf-8') as rf:
        csv_reader = csv.reader(rf)
        weights = []
        for line in csv_reader:
            weights.append(line)
        return np.array(weights)

    
def write_result(result, start_datetime, area, team, file_path='autosave.csv'):
    with open(file_path, 'a+', encoding='utf-8', newline='') as wf:
        csv_writer = csv.writer(wf)
        for i in range(result.shape[0]):
            csv_writer.writerow([datetime_change(start_datetime, i), result[i], area, team])
            
            
def date_to_index(date_or_index):
    date_list = np.array(['09-29', '09-30', '10-01', '10-02', '10-03', '10-04', '10-05', '10-06', '10-07', '10-08', '10-09', '10-10', '10-11', '10-12'])
    if type(date_or_index) == str:
        return np.where(date_list == date_or_index)[0][0]
    elif type(date_or_index) == int:
        return date_list[date_or_index]
    else:
        return None