import numpy as np
from utils import load_load_data
from pyswarm import pso

def mycon(aorg_load, x):
    aorg_load_mean = aorg_load.mean()
    delta = aorg_load_mean * x - aorg_load
    ad = np.mean(np.abs(delta))
    return ad


def get_weights(file_name):
    org_load=load_load_data(file_name, '09-29', '10-13')
    corg_load =org_load.reshape([-1, 2])
    load = np.zeros([14784,1])
    for i in range(1,14785):  # 写入数据
        load[i-1][0]=corg_load[i-1][1]
    pro_load = np.zeros([154,96])
    for n in range(14):
        for i in range(11):
            for m in range(96):
                pro_load[i+n*11][m]=load[m+i*1344+n*96][0]
    aorg_load=pro_load
    aload=(aorg_load/np.tile(np.mean(aorg_load, axis=1, keepdims=True), (1,96)))


    lb = [0.5] * 96
    ub = [1.8] * 96
    weights = []
    for sample in range(154):
        xopt, fopt = pso(mycon, lb, ub, args=(aload[sample],),swarmsize=100, maxiter=50,omega=0.8,phip=0.8,phig=0.8)
        weights.append(xopt)
    weights = np.array(weights)
    w1 = np.zeros([14,96])
    for i in range(14):        
        w1[i,:]=np.array(np.mean(weights[11*i:10+11*i,:],0))
    return w1