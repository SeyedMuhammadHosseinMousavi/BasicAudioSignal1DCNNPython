# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:41:49 2023

@author: S. M. hossein mousavi
"""

from pathlib import Path
import librosa, librosa.display
import numpy
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as ne
import sklearn.naive_bayes as nb
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import warnings

# Suppressing All Warnings
warnings.filterwarnings("ignore")

# Read Ausio Signals
ClassA = [
    librosa.load(p)[0] for p in Path().glob('Data/A*.wav')
]
ClassB = [
    librosa.load(p)[0] for p in Path().glob('Data/B*.wav')
]
ClassC = [
    librosa.load(p)[0] for p in Path().glob('Data/C*.wav')
]

# Extract Ausio Features 
def extract_featuresSC(signal):
    return [
        librosa.feature.spectral_centroid(signal)[0],
    ]
def extract_featuresRO(signal):
    return [
        librosa.feature.spectral_rolloff(signal)[0],
    ]
def extract_featuresZCR(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0],
    ]

NoOfSampClass = 20;
# spectral centroid
CatA1 = numpy.array([extract_featuresSC(x) for x in ClassA])
CatA1=np.reshape(CatA1, (NoOfSampClass, 47))
CatB1 = numpy.array([extract_featuresSC(x) for x in ClassB])
CatB1=np.reshape(CatB1, (NoOfSampClass, 47))
CatC1 = numpy.array([extract_featuresSC(x) for x in ClassC])
CatC1=np.reshape(CatC1, (NoOfSampClass, 47))
# spectral rolloff
CatA2 = numpy.array([extract_featuresRO(x) for x in ClassA])
CatA2=np.reshape(CatA2, (NoOfSampClass, 47))
CatB2 = numpy.array([extract_featuresRO(x) for x in ClassB])
CatB2=np.reshape(CatB2, (NoOfSampClass, 47))
CatC2 = numpy.array([extract_featuresRO(x) for x in ClassC])
CatC2=np.reshape(CatC2, (NoOfSampClass, 47))
# zero crossing rate
CatA3 = numpy.array([extract_featuresZCR(x) for x in ClassA])
CatA3=np.reshape(CatA3, (NoOfSampClass, 47))
CatB3 = numpy.array([extract_featuresZCR(x) for x in ClassB])
CatB3=np.reshape(CatB3, (NoOfSampClass, 47))
CatC3 = numpy.array([extract_featuresZCR(x) for x in ClassC])
CatC3=np.reshape(CatC3, (NoOfSampClass, 47))

# Feature Fusion
FinalA = np.hstack((CatA1, CatA2, CatA3))
FinalB = np.hstack((CatB1, CatB2, CatB3))
FinalC = np.hstack((CatC1, CatC2, CatC3))
FinalFeatures = np.vstack((FinalA, FinalB, FinalC))

# Dataset Size
DataSize=len(FinalFeatures)

# Labeling for Classification
ClassLabel = np.arange(DataSize)
ClassLabel[:20]=0
ClassLabel[20:40]=1
ClassLabel[40:60]=2

# Data Assign to X and Y
X=FinalFeatures
Y=ClassLabel

# Data Split
Xtr, Xte, Ytr, Yte = ms.train_test_split(X, Y, test_size = 0.35)

# KNN Classifier
trAcc=[]
teAcc=[]
Ks=[]
for i in range(1,5):
    KNN = ne.KNeighborsClassifier(n_neighbors = i)
    KNN.fit(Xtr, Ytr)
    trAcc.append(KNN.score(Xtr, Ytr))
    teAcc.append(KNN.score(Xte, Yte))
    Ks.append(i)
    
# Naive Bayes Classifier
NB = nb.GaussianNB()
NB.fit(Xtr, Ytr)
NBtrAcc = NB.score(Xtr, Ytr)
NBteAcc = NB.score(Xte, Yte)

# 1DCNN Structure
NoOfFeatures = len (FinalFeatures[0])
NoOfClass = 3;
model = Sequential()
model.add(Conv1D(64, 2, activation="relu", input_shape=(NoOfFeatures,1)))
model.add(Dense(16, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(NoOfClass, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])
model.summary()

# 1DCNN Training
model.fit(Xtr, Ytr, batch_size=4,epochs=100, verbose=0)

# Train and Test Results
print ('KNN Train Accuracy is :')
print (trAcc[-1])
print ('KNN Test Accuracy is :')
print (teAcc[-1])

print('Naive Bayes Train Accuracy is :')
print (NBtrAcc)
print('Naive Bayes Test Accuracy is :')
print (NBteAcc)

# Train Evaluation 
acc = model.evaluate(Xtr, Ytr)
print("1-D CNN Train Loss:", acc[0])
print("1-D CNN Train Accuracy:", acc[1])

# Test Evaluation 
acc2 = model.evaluate(Xte, Yte)
print("1-D CNN Test Loss:", acc2[0])
print("1-D CNN Test Accuracy:", acc2[1])

# Test Prediction
pred = model.predict(Xte)
pred_y = pred.argmax(axis=-1)

# Test Confusion Matrix
cm = confusion_matrix(Yte, pred_y)
print(cm)
cm_display2 = ConfusionMatrixDisplay(cm).plot()

