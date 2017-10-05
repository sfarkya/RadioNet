# This is code for Modulation recogition
# created by Saurabh Farkya
# Required Libraries

import numpy as np
from scipy.io import loadmat
import random

def readMatfile(settype):
# code for loading data from the .mat file.
    modulation_name = ['8PSK','BPSK','GFSK','PAM4','QPSK','WBFM','CPFSK','QAM16','QAM64','AM-DSB']

    if settype == 'train':
        data = np.ones([1,2,128])
        y =  []
        #data = np.reshape(data,[1,3])
        for i in range(10):
            for snr in range(-20,20,2):
                file_path = '/media/saurabh/New Volume/RadioML/' + modulation_name[i] + '_SNR' + str(snr) + '.mat'
                print file_path
                temp_dict = loadmat(file_path)
                temp_data = np.array(temp_dict['Data'])
                temp_data = temp_data[:5000,:,:]
                print temp_data.shape, data.shape
                data = np.concatenate((data,temp_data))
                y.append(5000*[i])
        y = np.asarray(y)
        y = np.reshape(y,[1,1000000 ])

    elif settype == 'validation':
        data = np.ones([1,2,128])
        y =  []
        #data = np.reshape(data,[1,3])
        for i in range(10):
            for snr in range(-20,20,2):
                file_path = '/media/saurabh/New Volume/RadioML/' + modulation_name[i] + '_SNR' + str(snr) + '.mat'
                print file_path
                temp_dict = loadmat(file_path)
                temp_data = np.array(temp_dict['Data'])
                temp_data = temp_data[5001:5501,:,:]
                print temp_data.shape, data.shape
                data = np.concatenate((data,temp_data))
                y.append(500*[i])
        y = np.asarray(y)
        y = np.reshape(y,[1,100000])

    elif settype == 'test':
        data = np.ones([1,2,128])
        y =  []
        #data = np.reshape(data,[1,3])
        for i in range(10):
            for snr in range(-20,20,2):
                file_path = '/media/saurabh/New Volume/RadioML/' + modulation_name[i] + '_SNR' + str(snr) + '.mat'
                print file_path
                temp_dict = loadmat(file_path)
                temp_data = np.array(temp_dict['Data'])
                temp_data = temp_data[5500:6000,:,:]
                print temp_data.shape, data.shape
                data = np.concatenate((data,temp_data))
                y.append(500*[i])
        y = np.asarray(y)
        y = np.reshape(y,[1,100000])

    #y = y[1:]

    data = data[1:,:,:]
    return data, y

    '''
    elif settype == 'test':
    elif settype == 'validation':
    '''
