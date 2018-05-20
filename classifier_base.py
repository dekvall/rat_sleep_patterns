#!usr/bin/python3
"""
David Ekvall & Rebecka Winqvist, 2018
Kandidatexamensarbete, Elektroteknik, KTH
"""

from scipy import stats,signal
import numpy as np
import scipy.io as sio
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.utils import plot_model

import random
from matplotlib import pyplot as plt

def makeNConsecutive(a, n):
    """Puts n features in a column"""
    b = np.zeros(((2*n+1)*a.shape[0], a.shape[1]-n*2))

    for idx in range(n, a.shape[1]-n):
        prevmat = a[:,[i for i in range (idx-n,idx)]]
        prevvec = np.transpose(prevmat.flatten("F"))
        currvec = a[:,idx]
        nexmat = a[:,[i for i in range (idx,idx+n)]]
        nexvec = np.transpose(nexmat.flatten("F"))
        temp = np.hstack((prevvec, currvec, nexvec))
        b[:,idx-n] = temp
    return b

def removeNFirstAndLastCol(a, n):
    """n is number of columns removed on both sides"""
    b = a;
    for _ in range(n):
        r, c = b.shape
        b = np.delete(b, c-1, 1)
        b = np.delete(b, 0, 1)
    return b
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def take_mean_and_subtract(row):
    mean = np.mean(row)
    temp = []
    for el in row:
        temp.append(el-mean)
    return np.array(temp)
    
def make_rowmean_zero(data_matrix):
    """Takes the mean of a row and subtracts it from
    every element of the row"""
    return np.apply_along_axis(take_mean_and_subtract, 1,
                               data_matrix)

def getNumberRelMaxMin(a):
    mins = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    maxs = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
    return np.sum(mins)+np.sum(maxs)

def extractFeatures(data):
    """Returns a 8 x datalength array of features"""
    EEG = data[0:2000]
    EMG = data[2000:4000]

    extracted_EEG = []
    extracted_EMG = []
    for col in EEG.T:
        temp = []
        temp.append(np.std(col))
        zero_crossings = len(np.where(np.diff(np.sign(col)))[0])
        temp.append(zero_crossings)
        temp.append(stats.skew(col))
        temp.append(stats.kurtosis(col))
        temp.append(getNumberRelMaxMin(col))#
        extracted_EEG.append(temp)
        
    for col in EMG.T:
        temp = []
        temp.append(np.std(col))
        zero_crossings = len(np.where(np.diff(np.sign(col)))[0])
        temp.append(zero_crossings)
        temp.append(stats.skew(col))
        temp.append(stats.kurtosis(col))
        temp.append(getNumberRelMaxMin(col))#
        extracted_EMG.append(temp)
    
    res1 = np.array(extracted_EEG)
    res2 = np.array(extracted_EMG)
    result = np.hstack((res1,res2))
    return result.T

def makeTrainingAndTestData(data, labels):
    """Returns training data in the format (datalength x 8) and (3 x datalength)"""
    trainingSamples = int(0.95*data.shape[1]); #This gives 95 % testdata
    indices = np.random.permutation(data.T.shape[0])
    training_idx, test_idx = indices[:trainingSamples], indices[trainingSamples:]
    trainingX, testX = data.T[training_idx,:], data.T[test_idx,:]
    trainingY, testY = labels.T[training_idx,:], labels.T[test_idx,:]
    return trainingX, trainingY, testX, testY
    
def createAndSaveModel(trainingX, trainingY, testX, testY):
    """Creates and returns a network model"""
    model = Sequential()
    
    model.add(Dense(15, activation='relu',input_dim=10))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(trainingX, trainingY, batch_size=8,
              epochs=20, verbose=1)
    metrics = model.evaluate(testX, testY, verbose=1)
    print(f'Model evaluated with an accuracy score of {round(metrics[1]*100,2)}%')
    return model

def createAndSaveConsecutiveModel(trainingX, trainingY, testX, testY):
    """Creates a consecutive model and returns it"""
    sz = trainingX.shape[1]
    model = Sequential()
    
    model.add(Dropout(0.2, input_shape = (sz,))) #input dropout
    model.add(Dense(int(sz*1.3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(sz*0.4), activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(3, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(trainingX, trainingY, batch_size=8,
              epochs=20, verbose=1)

    metrics = model.evaluate(testX, testY, verbose=1)
    print(f'Model evaluated with an accuracy score of {round(metrics[1]*100,2)}%')
    return model

def getStringState(row):
    idx_state = np.argmax(row)
    state = ""
    if idx_state == 0:
        state = "WAKE"
    elif idx_state == 1:
        state = "SWS"
    elif idx_state == 2:
        state = "REM"
    return state

def getStringPrediction(prediction):
    conf = np.amax(prediction)
    state = getStringState(prediction)
    return f'I am {round(conf*100,2)}% certain the rat is in a {state} state.'

def predictState(model, dataIn, verbose = True):
    #Keras takes data in shape: (number_of_samples_to_predict, input_shape)
    #Always one extra dimension to take care of
    new = np.array([dataIn])
    prediction = model.predict(new, batch_size = None)
    if verbose:
        print(getStringPrediction(prediction))
    return prediction
    
if __name__ == '__main__':
    #EDIT THIS CODE IN THE EVENT THAT YOU WANT TO ADD NEW MODELS
    contents = sio.loadmat("rat1.mat")
    data = contents["EEGandEMG"]
    labels = contents["Labels"]
    features = extractFeatures(data)

    mean_zero = make_rowmean_zero(features)
    norm_features = np.apply_along_axis(normalize, 1, mean_zero)

    CONSECUTIVE = False #Set to True if n != 0

    #It is possible to create models with any N
    #but rat_classifier.py currently only works with N = 5 and N = 10
    N = 5 

    if CONSECUTIVE:
        norm_consecutive_features = makeNConsecutive(norm_features,N)
        labels_consecutive = removeNFirstAndLastCol(labels,N)
        
        trainingX, trainingY, testX, testY = makeTrainingAndTestData(norm_consecutive_features, labels_consecutive)
        model = createAndSaveConsecutiveModel(trainingX, trainingY, testX, testY)
    else:
        trainingX, trainingY, testX, testY = makeTrainingAndTestData(norm_features, labels)
        model = createAndSaveModel(trainingX, trainingY, testX, testY)

    model.save("mod1_0.h5") #Needs to be named "mod{rat_number}_{N}.h5 and then placed in the corrct folder to work


