from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
from sklearn.preprocessing import LabelEncoder
from woa import jfs #importing woa whale feature selection class
import time
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pyswarms as ps #swarm package for PSO features selection algorithm
from SwarmPackagePy import testFunctions as tf
from sklearn import linear_model

main = tkinter.Tk()
main.title("META-HEURISTIC OPTIMIZATION")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global filename, dataset
global X, Y, XX
accuracy = []
precision = []
recall = []
fscore = []

lr_classifier = linear_model.LogisticRegression(max_iter=1000)
    
def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")
    dataset = pd.read_csv("Dataset/WPBC.csv")
    text.insert(END,str(dataset.head()))
    label = dataset.groupby('diagnosis').size()
    label.plot(kind="bar")
    plt.title("Total number of Benign & Malignant Cases found in dataset")
    plt.show()
    

def preprocessDataset():
    global X, Y, dataset, XX
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset.fillna(0, inplace = True)
    dataset['diagnosis'] = pd.Series(le.fit_transform(dataset['diagnosis'].astype(str)))
    text.insert(END,str(dataset.head()))
    dataset = dataset.values
    Y = dataset[:,1:2].ravel()
    X = dataset[:,2:dataset.shape[1]-1]
    XX = X
    
def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100    
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall    : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure  : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    

def runWhale():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    text.insert(END,"Total attributes/features found in dataset BEFORE applying Whale Optimization: "+str(X.shape[1])+"\n\n")
    start = time.time()
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, stratify=Y)
    fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}
    # parameter
    k    = 5     # k-value in KNN
    N    = 10    # number of particles
    T    = 100   # maximum number of iterations
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T}
    # perform feature selection
    fmdl = jfs(X, Y, opts)
    whale_sf   = fmdl['sf']
    X = X[:,whale_sf]
    end = time.time()
    text.insert(END,"Total attributes/features Selected from dataset AFTER applying Whale Optimization: "+str(X.shape[1])+"\n\n")
    text.insert(END,"Total time taken by Whale Optimization : "+str(end-start)+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    svm_cls = svm.SVC()
    svm_cls.fit(X_train,y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM with Whale Optimization", predict, y_test)

    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train,y_train)
    predict1 = dt_cls.predict(X_test)
    calculateMetrics("Decision Tree with Whale Optimization", predict1, y_test)

    knn_cls = KNeighborsClassifier(n_neighbors=2)
    knn_cls.fit(X_train,y_train)
    predict = knn_cls.predict(X_test)
    calculateMetrics("KNN with Whale Optimization", predict, y_test)
    LABELS = ['Benign', 'Malignant'] 
    conf_matrix = confusion_matrix(y_test, predict1) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    


#PSO function to calculate importance of each features
def f_per_particle(m, alpha):
    global X
    global Y
    global lr_classifier
    total_features = 30
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    lr_classifier.fit(X_subset, Y)
    P = (lr_classifier.predict(X_subset) == Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def fun(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


def runPSO():
    global X, Y, XX
    X = XX
    text.insert(END,"Total attributes/features found in dataset BEFORE applying PSO: "+str(X.shape[1])+"\n\n")
    start = time.time()
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
    cost, pos = optimizer.optimize(fun, iters=2)#OPTIMIZING FEATURES
    X_selected_features = X[:,pos==1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1
    end = time.time()
    text.insert(END,"Total attributes/features Selected from dataset AFTER applying PSO: "+str(X_selected_features.shape[1])+"\n\n")
    text.insert(END,"Total time taken by PSO: "+str(end-start)+"\n\n")

    X_train, X_test, y_train, y_test = train_test_split(X_selected_features, Y, test_size=0.2)
    svm_cls = svm.SVC()
    svm_cls.fit(X_train,y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM with PSO", predict, y_test)

    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train,y_train)
    predict1 = dt_cls.predict(X_test)
    calculateMetrics("Decision Tree with PSO", predict1, y_test)

    knn_cls = KNeighborsClassifier(n_neighbors=2)
    knn_cls.fit(X_train,y_train)
    predict = knn_cls.predict(X_test)
    calculateMetrics("KNN with PSO", predict, y_test)

    LABELS = ['Benign', 'Malignant'] 
    conf_matrix = confusion_matrix(y_test, predict1) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

def graph():
    df = pd.DataFrame([['Whale SVM','Precision',precision[0]],['Whale SVM','Recall',recall[0]],['Whale SVM','F1 Score',fscore[0]],['Whale SVM','Accuracy',accuracy[0]],
                       ['Whale Decision Tree','Precision',precision[1]],['Whale Decision Tree','Recall',recall[1]],['Whale Decision Tree','F1 Score',fscore[1]],['Whale Decision Tree','Accuracy',accuracy[1]],
                       ['Whale KNN','Precision',precision[2]],['Whale KNN','Recall',recall[2]],['Whale KNN','F1 Score',fscore[2]],['Whale KNN','Accuracy',accuracy[2]],
                       ['PSO SVM','Precision',precision[3]],['PSO SVM','Recall',recall[3]],['PSO SVM','F1 Score',fscore[3]],['PSO SVM','Accuracy',accuracy[3]],
                       ['PSO Decision Tree','Precision',precision[4]],['PSO Decision Tree','Recall',recall[4]],['PSO Decision Tree','F1 Score',fscore[4]],['PSO Decision Tree','Accuracy',accuracy[4]],
                       ['PSO KNN','Precision',precision[5]],['PSO KNN','Recall',recall[5]],['PSO KNN','F1 Score',fscore[5]],['PSO KNN','Accuracy',accuracy[5]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("Whale & PSO Accuracy, Precision, Recall & FScore Graph")
    plt.show()
    

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='META-HEURISTIC OPTIMIZATION ALGORITHMS BASED FEATURE SELECTION FOR CLINICAL BREAST CANCER DIAGNOSIS')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload WPBC Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

whaleButton = Button(main, text="Run SVM, Decision Tree & KNN with Whale Optimization", command=runWhale)
whaleButton.place(x=50,y=200)
whaleButton.config(font=font1)

psoButton = Button(main, text="Run SVM, Decision Tree & KNN with PSO", command=runPSO)
psoButton.place(x=50,y=250)
psoButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=350)
exitButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=75)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
