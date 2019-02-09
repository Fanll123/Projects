# Import the data
import pandas as pd
dataset = "C:/Users/Zer0/Desktop/drug_consumption.data"
drug_df = pd.read_table(dataset, sep=",", header=None, usecols=[1,2,3,6,7,8,9,10,11,12,18,20,22,23,24]
                        , names=['Age','Gender','Education','Nscore',
                                 'Escore','Oscore','Ascore','Cscore','Impulsive',
                                 'Sensation','Cannabis','Cocaine','Ecstasy','Heroin',
                                  'Ketamine'])

import warnings
warnings.filterwarnings('ignore')

# will show all the columns
pd.set_option('display.max_columns', None)
drug_df.head()

############################################################
# Data cleaning
############################################################
import numpy as np
import imblearn

# check features distribution
import seaborn as sns
sns.distplot(drug_df['Age'], kde=True)
sns.distplot(drug_df['Gender'], kde=True)
sns.distplot(drug_df['Nscore'], kde=True)

# check correlations between all the features
corr = drug_df[['Age','Gender','Education','Nscore',
                'Escore','Oscore','Ascore','Cscore','Impulsive',
                'Sensation']].corr()

# show heapmap of correlations
sns.heatmap(corr)

# Select level CL0 and CL1 as Low Drug Risk  
y = np.where(drug_df['Cannabis']=='CL0',1,0) + np.where(drug_df['Cannabis']=='CL1',1,0)

# Drop unused columns
to_drop = ['Cannabis','Cocaine','Ecstasy','Heroin','Ketamine']
feat_space = drug_df.drop(to_drop, axis=1)
X = feat_space

# Standarization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

################################################################
# Training and Evaluation
################################################################ 
# Splite data into training and testing
from sklearn import model_selection

# Reserve 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

print('training data has %d observation with %d features'% X_train.shape)
print('test data has %d observation with %d features'% X_test.shape)

#@title build models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 

# Logistic Regression
classifier_logistic = LogisticRegression()

# K Nearest Neighbors
classifier_KNN = KNeighborsClassifier()

# Random Forest
classifier_RF = RandomForestClassifier()

# SVM model
classifier_SVC = SVC()

# Train the model
classifier_logistic.fit(X_train, y_train)

# Prediction of test data
classifier_logistic.predict(X_test)
# Accuracy of test data
classifier_logistic.score(X_test, y_test)


# Use 5-fold Cross Validation to get the accuracy for different models
model_names = ['Logistic Regression','KNN','Random Forest']
model_list = [classifier_logistic, classifier_KNN, classifier_RF]
count = 0

for classifier in model_list:
    cv_score = model_selection.cross_val_score(classifier, X_train, y_train, cv=5)
    # cprint(cv_score)
    print('Model accuracy of %s is: %.3f'%(model_names[count],cv_score.mean()))
    count += 1

cv_score = model_selection.cross_val_score(classifier_SVC, X_train, y_train, cv=5)
print('Model accuracy of SVM is: %.3f'%(cv_score.mean()))

# Grid search for model optimization
from sklearn.model_selection import GridSearchCV

# helper function for printing out grid search results 
def print_grid_search_metrics(gs):
    print ("Best score: %0.3f" % gs.best_score_)
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Possible hyperparamter options for Logistic Regression Regularization
# Penalty is choosed from L1 or L2
# C is the lambda value(weight) for L1 and L2
parameters = {
    'penalty':('l1', 'l2'), 
    'C':(1, 5, 10)
}
Grid_LR = GridSearchCV(LogisticRegression(),parameters, cv=5)
Grid_LR.fit(X_train, y_train)

# the best hyperparameter combination
print_grid_search_metrics(Grid_LR)

# best model
best_LR_model = Grid_LR.best_estimator_

# Possible hyperparamter options for KNN
# Choose k
parameters = {'n_neighbors':[3,5,7,10] }
Grid_KNN = GridSearchCV(KNeighborsClassifier(),parameters, cv=5)
Grid_KNN.fit(X_train, y_train)
# best k
print_grid_search_metrics(Grid_KNN)

# Possible hyperparamter options for Random Forest
# Choose the number of trees
parameters = {
    'n_estimators' : [40,60,80]
}
Grid_RF = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
Grid_RF.fit(X_train, y_train)

# best number of tress
print_grid_search_metrics(Grid_RF)
# best random forest
best_RF_model = Grid_RF.best_estimator_


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# calculate accuracy, precision and recall
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print (classifier)
    print ("Accuracy is: %0.3f" % accuracy)
    print ("precision is: %0.3f" % precision)
    print ("recall is: %0.3f" % recall)

# print out confusion matrices
def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not','Churn']
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, interpolation='nearest',cmap=plt.get_cmap('Reds'))
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

%matplotlib inline

# Confusion matrix, accuracy, precison and recall for random forest and logistic regression
confusion_matrices = [
    ("Random Forest", confusion_matrix(y_test,best_RF_model.predict(X_test))),
    ("Logistic Regression", confusion_matrix(y_test,best_LR_model.predict(X_test))),
]

draw_confusion_matrices(confusion_matrices)

from sklearn.metrics import roc_curve
from sklearn import metrics

# Use predict_proba to get the probability results of Random Forest
y_pred_rf = best_RF_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

# ROC curve of Random Forest result
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - RF model')
plt.legend(loc='best')
plt.show()

from sklearn import metrics

# AUC score
metrics.auc(fpr_rf,tpr_rf)