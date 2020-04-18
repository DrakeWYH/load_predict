from data_prepare import data_prepare
from model_train import model_train_guangzhou
from model_predict import model_predict_guangzhou

data_prepare()
model_train_guangzhou()
model_predict_guangzhou('09-29')
model_predict_guangzhou('09-30')