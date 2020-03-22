# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
## importing the required libraries for analysis

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

## loading the dataset as a dataframe using pandas

Training_File = "C:\\Users\\JMSch\\Documents\\MEGA\\05_Paris\\ESCP Business School\\MSc Big Data & Business Analytics\\Machine Learning in Python\\train.csv"
Train = pd.read_csv(Training_File)


# %%
## initial eye-balling of the columns available in our dataset

Train.columns


# %%
print(Train["label"].mean())
print(Train["label"].min())
print(Train["label"].max())


# %%
## the variable to be predicted is label; either it is a sale or not

label = Train["label"].values


# %%
Train["label"].value_counts()
sns.countplot(x='label',data=Train,palette='hls')
plt.show()
plt.savefig('count_plot')


# %%
## from the original dataset we drop the variables that have no indicative nature for our predictor

## Train_Clean = Train.drop(columns=['label','id','visitTime','purchaseTime','C1','C3','C10']).values


# %%
## we select the X(input variables) and the y(outcome variable --> sale or not)
## X = Train_Clean.copy()
## y = np.array(label.copy())


# %%
## since we are dealing with an unbalanced dataset (far more non-sales than sales), we need to create a sample where this is 

id_toTrain = np.array([np.where(Y_train==i)[0] for i in range(nb_class)])

size_max = [len(id_toTrain[i]) for i in range(nb_class)]
print(f"Before resampling we the distribution is as follows [non-sale, sale]: {size_max}")

blc = 150
for i in range(len(size_max)):
    if size_max[i] > blc:
        size_max[i] = int(blc*(np.log10(size_max[i]/blc)+1))
    else:
        size_max[i] = int(blc/(np.log10(blc/size_max[i])+1))

print(f"After resampling we the distribution is as follows [non-sale, sale]: {size_max}")

for i in range(nb_class):
    if len(id_toTrain[i]) > size_max[i]:
        id_toTrain[i], tmp = train_test_split(id_toTrain[i], test_size=1-size_max[i]/len(id_toTrain[i]))
    else:
        id_toTrain[i] = np.concatenate((id_toTrain[i], id_toTrain[i][np.random.randint(len(id_toTrain[i]), size=int(size_max[i]-len(id_toTrain[i])))]))
id_toTrain = np.concatenate(id_toTrain)
X_toTrain = X_train[id_toTrain]
Y_toTrain = Y_train[id_toTrain]


# %%
xgb.plot_importance(clf, importance_type = 'weight', max_num_features=15)
plt.show()


# %%
len(Train['label'])


# %%
## With our training data created, I’ll up-sample the no-subscription using the SMOTE algorithm(Synthetic Minority Oversampling Technique). At a high level, SMOTE:
## Works by creating synthetic samples from the minor class (no-subscription) instead of creating copies.
## Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations


X = Train.loc[:, Train.columns != 'label']
y = Train.loc[:, Train.columns == 'label']
y=y.astype('int')
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['label'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['label']==-1]))
print("Number of subscription",len(os_data_y[os_data_y['label']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['label']==-1])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['label']==1])/len(os_data_X))


# %%
os_data_y['label'].mean()


# %%
Train = pd.concat([os_data_y, os_data_X], axis=1)
Train


# %%
## Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and choose either the best or worst performing feature, setting the feature aside and then repeating the process with the rest of the features. This process is applied until all features in the dataset are exhausted. The goal of RFE is to select features by recursively considering smaller and smaller sets of features.

Train_final_vars = Train.columns.values.tolist()
y=['label']
X=[i for i in Train_final_vars if i not in y]

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
print(columns)


# %%
## save only the variables that have an a significant (P<0.05) effect on the outcome variable
cols=['N1', 'N3', 'N8','N9', 'N10']

X=os_data_X[cols]
y=os_data_y['label']
os_data_y['label'].replace({-1: 0}, inplace=True)

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# %%
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# %%
y_pred = logreg.predict(X)

pred_list = y_pred.tolist()

X_with_prediction = X
X_with_prediction ["Predicted score"] = pred_list

X_with_prediction ['']


# %%
Train['id']


# %%
Export = pd.concat([Train['id'], X_with_prediction['Predicted score']], axis=1)
Export.to_csv(r'C:\Users\JMSch\Documents\MEGA\05_Paris\ESCP Business School\MSc Big Data & Business Analytics\Machine Learning in Python\Final assignment - Scheepers, J.M..csv', index = False)


# %%
confusion_matrix = confusion_matrix(X_test, y_pred)
print(confusion_matrix)


# %%
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

