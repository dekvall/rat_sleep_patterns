#!usr/bin/python3
"""
David Ekvall & Rebecka Winqvist, 2018
Kandidatexamensarbete, Elektroteknik, KTH
"""
from classifier_base import *
import sys
import time

MODEL_NUMS = 1,2,3,4,5

class RatClassifier():
    def __init__(self):
        self.modelsN0 = []
        self.modelsN5 = []
        self.modelsN10 = []
        self.data = None
        self.features = None

    def addModelFromFile(self, path, n=0):
        if n == 0:
            self.modelsN0.append(load_model(path))
        if n == 5:
            self.modelsN5.append(load_model(path))
        if n == 10:
            self.modelsN10.append(load_model(path))
    
    def loadData(self, path, varname = "EEGandEMG"):
        """path must reference a 4000 x N matlab array
            with a name of "EEGandEMG", N is the number of epochs"""
        contents = sio.loadmat(path)
        self.data = contents[varname]
        self.createFeatureArray()
    
    def createFeatureArray(self):
        raw_features = extractFeatures(self.data)
        mean_zero = make_rowmean_zero(raw_features)
        self.features = np.apply_along_axis(normalize, 1, mean_zero)
        
    def classifyWithAllModels(self):
        print("Classifying with N = 0")
        self.predictWithN0(False)
        print("Classifying with N = 5")
        self.predictWithN5()
        print("Classifying with N = 10")
        self.predictWithN10()

        result = self.resultN0 + self.resultN5 + self.resultN10

        labels = np.zeros_like(result)
        labels[np.arange(len(result)), result.argmax(1)] = 1

        d = dict()
        d["Labels"] = labels.T
        sio.savemat(sys.argv[2], d)
        
    def predictWithN0(self, alone=True, verbose = False):
        if verbose:
            print("Predicting with n = 0")
        preds = None
        if self.data is None:
            print("You have to load data first")
            return
        for col in self.features.T:
            pred = np.array([0.0,0.0,0.0])
            for m in self.modelsN0:
                pred = pred + predictState(m, col, verbose)
            if preds is None:
                preds = pred
            else:
                preds = np.vstack((preds,pred))
        self.resultN0 = preds

    def predictWithN5(self, verbose = False):
        if self.data is None:
            print("You have to load data first")
            return
        #Unable to classify the first five epochs
        initial = np.array([0.0, 0.0, 0.0]) 
        initial = np.vstack([initial]*5)
        preds = initial
        features_N5 = makeNConsecutive(self.features, 5)
        for col in features_N5.T:
            pred = np.array([0.0,0.0,0.0])
            for m in self.modelsN5:
                pred = pred + predictState(m, col, verbose)
            preds = np.vstack((preds,pred))
        preds = np.vstack((preds,initial))
        self.resultN5 = preds
        
    def predictWithN10(self, verbose = False):
        if self.data is None:
            print("You have to load data first")
            return
        #Unable to classify the first ten epochs
        initial = np.array([0.0, 0.0, 0.0]) 
        initial = np.vstack([initial]*10)
        preds = initial
        features_N10 = makeNConsecutive(self.features, 10)
        for col in features_N10.T:
            pred = np.array([0.0,0.0,0.0])
            for m in self.modelsN10:
                pred = pred + predictState(m, col, verbose)
            preds = np.vstack((preds,pred))
        preds = np.vstack((preds,initial))
        self.resultN10 = preds

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Start the program like so:\n\
python rat_classifier.py *data_to_classify* *file_to_save_labels*\n\
EX: python rat_classifier.py rat8.mat labels8.mat")
        exit(0)
    c = RatClassifier()
    for i in MODEL_NUMS:
        c.addModelFromFile(f"models/n0/mod{i}_0.h5")
        c.addModelFromFile(f"models/n5/mod{i}_5.h5",5)
        c.addModelFromFile(f"models/n10/mod{i}_10.h5",10)
    p = sys.argv[1]
    start = time.time()
    c.loadData(p)
    c.classifyWithAllModels()
    print("Done! Elapsed time:",time.time()-start, "seconds.")
