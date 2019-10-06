# Rat classification
A program that takes EEG and EMG data to label the sleep state of rats.
REQUIRES `python 3.6+`.

25 hours of data takes roughly 4 minutes to classify, depending on platform.

## Paper
Paper available [here](http://kth.diva-portal.org/smash/record.jsf?pid=diva2:1264829)

## Dependencies
```
pip install --upgrade tensorflow
pip install keras
pip install numpy
pip install scipy
pip install matplotlib
pip install h5py
```
## How it works
Start the program like so:
```
python rat_classifier.py *data_to_classify* *path_to_save_labels*
EX:
python rat_classifier.py rat8.mat labeled8.mat
```
data_to_classify must contain a `4000 x N` array named EEGandEEG on the form
```
EEGandEMGColumn = [EEG;
                   EMG]
```
where EEG and EMG both are 2000 x 1 vectors.

The labels will be saved in the path_to_save_labels in a `3 x N` array named Labels
The columns of the array will be one hot encoded like this
```
LabelColumn = [WAKE;
              SWS;
              REM]
or
LabelColumn = [0;
               0;
               1]
means that the rat is in REM sleep
```
The program uses a voting system, fifteen models will vote with their predictions and the sleep state with the highest percentage in total will "win". This makes the program more likely to correctly classify unknown cases.
## Models
15 models can be found in the folder "models".
However, more models can be added by poking around in the base_classifier.py code.
Also don't forget to add its number to the MODEL_NUMS global in rat_classifier.py
More models should mean better classification accuracy
## Performance
At the moment this generalized classifier performs as follows on these rats.
```
Rat1: 96.25 %
Rat2: 91.31 %
Rat3: 93.27 %
Rat4: 87.56 %
Rat5: 86.40 %
```
