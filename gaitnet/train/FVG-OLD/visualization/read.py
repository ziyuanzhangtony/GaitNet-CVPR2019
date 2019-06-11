


import scipy.io as sio
#
# sio.loadmat('data/data_BG.pickle')

import pickle
key = 'WS'

file_ = open('data/data_%s.pickle'%key,'rb')
# file_CB = open('data/data_CB.pickle','rb')
# file_CL = open('data/data_CL.pickle','rb')
# file_WS = open('data/data_WS.pickle','rb')
data = pickle.load(file_)
# data[0] = data[0].cpu().data.numpy()
# data[1] = data[1].cpu().data.numpy()
dict = {
    'glr_%s'%(key):data[0],
    'prb_%s'%(key):data[1],
    'gt_%s'%(key):data[2]
}
sio.savemat('data/data_%s.mat'%(key),dict)
