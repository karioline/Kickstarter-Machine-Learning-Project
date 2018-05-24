# -*- coding: utf-8 -*-
"""
Kari Shelton
CMPS 570
Machine Learning Project

MODEL:          Random Forest Decision Tree with Threefold Split (Tr, Val, Ts)

Help from 'Introduction to Machine Learning with Python' by Muller and Guido
pp. 264-265
	
Truth:
    0 = failed = Negative Class
    1 = successful = Positive Class
"""

import pandas as pd
import numpy as np

ks_data_all = pd.read_csv("ks_data.csv")
data = ks_data_all[['main_category', 'deadline_month', 'launched_month', 'Length_of_fundraising', 'backers', 'country', 'usd_goal']]
target = pd.read_csv("ks_target.csv")


#one-hot-encoding ... aka get_dummies
data_dummies = pd.get_dummies(data)


# mglearn p.219
"""-------------- Get Data in Right Format for Machine Learning --------------------""" 

# extract features
features = data_dummies.loc[:, 'Length_of_fundraising' : 'country_US']
X = features.values
# extract true target

y = target.values
c, r = y.shape
y = y.reshape(c,)  #https://stackoverflow.com/questions/31995175/scikit-learn-cross-val-score-too-many-indices-for-array


"""---------------------------------------------------------"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time

start_time = time()

val_ts_size = int(float(0.2 * len(X)))

#Split X into tr+val set and ts set
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0, test_size=val_ts_size)
#Split X_trainval set into tr and val sets
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=0, test_size=val_ts_size)

print("Size of training set: {}\nSize of validation set: {}\nSize of test set: {}\n".format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))

best_score = 0
scores = [0,0]
best_depth = {}

tuning_time = time()

for depth in [5,10,15,20,25,30]:
    forest = RandomForestClassifier(max_depth=depth,random_state=0,)
    forest.fit(X_train, y_train)
    # score returns: the mean accuracy on the given test data and labels.
    acc = forest.score(X_val, y_val)
    scores = np.vstack((scores, [depth, acc]))
    #if we got a better score, store the score and tree depth parameter
    if acc > best_score:
        best_score = acc
        best_depth = {'max_depth' : depth}

total_tuning_time = ("\nTuning the parameters took %.2f minutes. \n" % ((time() - tuning_time)/60))
print(total_tuning_time)

#will use depth_score to create an epoch-like error curve
scores = np.delete(scores, 0,0)

#rebuild a model on the combined training and validation set, and evaluate it on the test set
forest = RandomForestClassifier(**best_depth,random_state=0)
forest.fit(X_trainval, y_trainval)
test_score = forest.score(X_test, y_test)

pred_y = forest.predict(X_test)


"""--------------------------MODEL PERFORMANCE MEASURES------------------------------"""

#function below returns sensitivity value
def sens(tp, fp, tn, fn):
    return tp / (tp+fn)

#function below returns specificity
def spec(tp, fp, tn, fn):
    return tn / (tn+fp)
    
#function below returns accuracy
def acc(tp, fp, tn, fn):
    return (tp+tn) / (tn+fp+fn+tp)

#function below returns precision
def prec(tp, fp, tn, fn):
    return tp/(tp + fp)

#function below returns F1-Score
def f1_score(prec, sens):
    return (2 * prec * sens)/(prec + sens) 


# Positive class = successful = 1
# Negative class = failed = 0
def model_perf_measure(y_actual, y_pred):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(len(y_pred)): 
        if (y_actual[i]==1) and (y_pred[i]==1):
           TP += 1
        if (y_actual[i]==1) and (y_pred[i]==0):
           FN += 1
        if (y_actual[i]==0) and (y_pred[i]==1):
           FP += 1
        if (y_actual[i]==0) and (y_pred[i]==0):
           TN += 1
    
    sensitivity = sens(TP, FP, TN, FN)
    specificity = spec(TP, FP, TN, FN)
    accuracy = acc(TP, FP, TN, FN)
    precision = prec(TP, FP, TN, FN)
    f1 = f1_score(precision, sensitivity)
    fpr, tpr, thresholds = metrics.roc_curve(y_actual, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    perf = ['TP', 'FP', 'TN', 'FN', 'sensitivity', 'specificity', 'accuracy', 'f1', 'AUC']
        
    return(np.stack((perf, [TP, FP, TN, FN, sensitivity, specificity, accuracy, f1, auc])))


perf = model_perf_measure(y_test, pred_y)
print(perf.T)

"""-------------------------- Epoch-like Error Curve ----------------------------"""
import matplotlib.pyplot as plt


plt.figure(1)
plt.plot(scores.T[0],scores.T[1])
plt.xlabel('Tree Depth')
plt.xticks([5,10,15,20,25,30])
plt.ylabel('Accuracy')
plt.title('Epoch Error Curve: rfdt_threefold')
plt.show()




"""-------------------------- Threshold Work -----------------------------"""


u = forest.predict_proba(X_test)
t_0 = []
t_25 = []
t_5 = []
t_75 = []
t_1 = []

for i in range(0,66335):
   if (u[i][0] > 0.0):
       t_0.append(0)
   else:
       t_0.append(1)

for i in range(0,66335):
   if (u[i][0] > 0.25):
       t_25.append(0)
   else:
       t_25.append(1)
       
for i in range(0,66335):
   if (u[i][0] > 0.5):
       t_5.append(0)
   else:
       t_5.append(1)

for i in range(0,66335):
   if (u[i][0] > 0.75):
       t_75.append(0)
   else:
       t_75.append(1)

for i in range(0,66335):
   if (u[i][0] > 1.0):
       t_1.append(0)
   else:
       t_1.append(1)       
    


def calculate_contingency(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)): 
        if (y_actual[i]==1) and (y_pred[i]==1):
           TP += 1
        if (y_actual[i]==1) and (y_pred[i]==0):
           FN += 1
        if (y_actual[i]==0) and (y_pred[i]==1):
           FP += 1
        if (y_actual[i]==0) and (y_pred[i]==0):
           TN += 1
    return (TP, FP, TN, FN)


cm_0 = calculate_contingency(y_test, np.asarray(t_0))
cm_25 = calculate_contingency(y_test, np.asarray(t_25))
cm_5 = calculate_contingency(y_test, np.asarray(t_5))
cm_75 = calculate_contingency(y_test, np.asarray(t_75))
cm_1 = calculate_contingency(y_test, np.asarray(t_1))




"""-------------------------- ROC Curve -----------------------------"""

from sklearn.metrics import roc_curve, auc

#model's
fpr, tpr, thresholds = roc_curve(y_test, pred_y, pos_label=1)
roc_auc = auc(fpr, tpr) 

#threshold 0.0 
fpr_0, tpr_0, thresholds_0 = roc_curve(y_test, np.asarray(t_0), pos_label=1)
roc_auc_0 = auc(fpr_0, tpr_0) 

#threshold 0.25
fpr_25, tpr_25, thresholds_25 = roc_curve(y_test, np.asarray(t_25), pos_label=1)
roc_auc_25 = auc(fpr_25, tpr_25)

#threshold 0.5
fpr_5, tpr_5, thresholds_5 = roc_curve(y_test, np.asarray(t_5), pos_label=1)
roc_auc_5 = auc(fpr_5, tpr_5)

#threshold 0.75 
fpr_75, tpr_75, thresholds_75 = roc_curve(y_test, np.asarray(t_75), pos_label=1)
roc_auc_75 = auc(fpr_75, tpr_75)

#threshold 1.0
fpr_1, tpr_1, thresholds_1 = roc_curve(y_test, np.asarray(t_1), pos_label=1)
roc_auc_1 = auc(fpr_1, tpr_1)



#rates holds all of the threshold points
rates = np.asarray(([fpr_0[0],fpr_0[0]], [fpr_25[1],tpr_25[1]], [fpr[1],tpr[1]], [fpr_5[1],tpr_5[1]], [fpr_75[1],tpr_75[1]], [fpr_1[1],tpr_1[1]]))

plt.figure(2)
plt.plot(rates.T[0], rates.T[1], 'k-', label='Model\'s auc = %0.2f' % (roc_auc))
plt.plot(fpr, tpr, 'ko', label='Model\'s')
plt.plot(fpr_0, tpr_0, 'bo', label='Threshold 0.0')
plt.plot(fpr_25, tpr_25, 'go', label='Threshold 0.25')
plt.plot(fpr_5, tpr_5, 'ro', label='Threshold 0.5')
plt.plot(fpr_75, tpr_75, 'co', label='Threshold 0.75')
plt.plot(fpr_1, tpr_1, 'yo', label='Threshold 1.0')
#guess line
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: rfdt_threefold')
plt.legend(loc="lower right")



"""---------------------Histogram of Contingency Values-----------------"""

# http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/
import matplotlib.cm as cm
import operator as o

dpoints = np.array([['threshold=0.0', 'TP', cm_0[0]],
                    ['threshold=0.0', 'FP', cm_0[1]],
                    ['threshold=0.0', 'TN', cm_0[2]],
                    ['threshold=0.0', 'FN', cm_0[3]],
                    ['threshold=0.25', 'TP', cm_25[0]],
                    ['threshold=0.25', 'FP', cm_25[1]],
                    ['threshold=0.25', 'TN', cm_25[2]],
                    ['threshold=0.25', 'FN', cm_25[3]],
                    ['threshold=0.5', 'TP', cm_5[0]],
                    ['threshold=0.5', 'FP', cm_5[1]],
                    ['threshold=0.5', 'TN', cm_5[2]],
                    ['threshold=0.5', 'FN', cm_5[3]],
                    ['threshold=0.75', 'TP', cm_75[0]],
                    ['threshold=0.75', 'FP', cm_75[1]],
                    ['threshold=0.75', 'TN', cm_75[2]],
                    ['threshold=0.75', 'FN', cm_75[3]],
                    ['threshold=1.0', 'TP', cm_1[0]],
                    ['threshold=1.0', 'FP', cm_1[1]],
                    ['threshold=1.0', 'TN', cm_1[2]],
                    ['threshold=1.0', 'FN', cm_1[3]],
                    ])

fig = plt.figure(3)
ax = fig.add_subplot(111)

def barplot(ax, dpoints):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.
    
    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''
    
    # Aggregate the conditions and the categories according to their
    # mean values
    conditions = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,0])]
    categories = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,1])]
    
    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]
    
    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

    # the space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))
    
    # Create a set of bars at each position
    for i,cond in enumerate(conditions):
        indeces = range(1, len(categories)+1)
        vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cond, 
               color=cm.Accent(float(i) / n))
    
    # Set the x-axis tick labels to be equal to the categories
    ax.set_xticks(indeces)
    ax.set_xticklabels(categories)
    #plt.setp(plt.xticks()[1])
    
    # Add the axis labels
    ax.set_ylabel("Number of Samples")
    #ax.set_xlabel("Contingency Values")
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.04,1), loc='upper left')
    
        
barplot(ax, dpoints)
plt.title("rfdt_threefold")
plt.show()


# print tabulated contingency comparison data
# https://stackoverflow.com/questions/41140647/python-printing-lists-with-tabulate

model_values = (int(float(perf.T[0][1])),int(float(perf.T[1][1])),int(float(perf.T[2][1])),int(float(perf.T[3][1])))
g = np.vstack((['TP', 'FP', 'TN', 'FN'],cm_0,cm_25,cm_5,cm_75,cm_1, model_values))
TP_values = np.delete(g.T[0],0,0)
FP_values = np.delete(g.T[1],0,0)
TN_values = np.delete(g.T[2],0,0)
FN_values = np.delete(g.T[3],0,0)
thrs = [0.0,0.25,0.5,0.75,1.0]
titles = ['thresholds', 'TP', 'FP', 'TN', 'FN']

print ('{:<6}|{:<6}|{:<6}|{:<6}|{:<6}'.format(*titles))
for item in zip(thrs, TP_values, FP_values, TN_values, FN_values):
    print ('{:<6}    |{:<6}|{:<6}|{:<6}|{:<6}'.format(*item))


total_execution_time = ("\nTotal execution time took %.2f minutes. \n" % ((time() - start_time)/60))
print(total_execution_time)