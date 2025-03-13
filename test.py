import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from woa import jfs
from sklearn.model_selection import train_test_split

le = LabelEncoder()

dataset = pd.read_csv("Dataset/WPBC.csv")
dataset.fillna(0, inplace = True)
dataset['diagnosis'] = pd.Series(le.fit_transform(dataset['diagnosis'].astype(str)))
dataset = dataset.values

Y = dataset[:,1:2].ravel()
X = dataset[:,2:dataset.shape[1]-1]
print(X.shape)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, stratify=Y)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of particles
T    = 100   # maximum number of iterations
opts = {'k':k, 'fold':fold, 'N':N, 'T':T}

# perform feature selection
fmdl = jfs(X, Y, opts)
sf   = fmdl['sf']

X = X[:,sf]

print(X.shape)

