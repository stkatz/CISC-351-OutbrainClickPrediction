"""
Project: Outbrain Click Prediction
Task: Predict which recommended content each user will click.
Group: 5
Group members: Analeeze Mendonsa, Samathan Katz, and Rachel Venis
Date: April 3rd, 2019 

Document Description: Uses the files manipulated in train_setup to run our classification models on it 
"""
# %% Run Modules 
"""Imports the necessary Libraries needed to run the models"""
import pandas as pd #Pandas: Module used for data manipulation and analysis
import numpy as np #Numpy: Module for scientific computing 
import seaborn as sb 
import matplotlib.pyplot as plt #MatPlotLib: Module for data visualization
# %matplotlib inline
#Sklearn: Module of easy-use data analysis models 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
#imblearn: Module to implement SMOTE
from imblearn.over_sampling import SMOTE
import pprint as pp #pprint: Module provides the ability to “pretty-print” arbitrary Python data structures

# %% Reads CSV Files
import pandas as pd
merge = pd.read_csv('MERGE.csv')
# %% Creating a Copy of the Merge File 
train = merge.copy()

# %% Prints column heads and first 10 values
train.head(10)

# %% Drop "Unnamed" Column and prints column heads and first 10 values
train.drop(['Unnamed: 0'],axis=1,inplace=True)
train.head(10)

# %% Drop "publish_time" Column and prints column heads and first 10 values
#train['publish_time'] = map(lambda x: x + 1465876799998, train['publish_time'].values)
train.drop(['publish_time'],axis=1,inplace=True)
train.head(10)

# %% Drops all the Null values 
train.dropna()

#%% Creates a copy of the train file with the data manipulation 
integer_encoded = train.copy()

# Runs Exploratory Data Analysis on each Column 
integer_encoded.describe()

# %% Splits the Train and Test set, where 30% of the data is test data and 70% is training data
from sklearn.model_selection import train_test_split
training_features, test_features, training_target, test_target = train_test_split(integer_encoded.drop(['clicked'], axis=1), integer_encoded['clicked'], test_size = 0.3, random_state=27)

x_train, x_val, y_train, y_val = train_test_split(training_features, training_target, test_size = .3, random_state=12, stratify= training_target)

# %% Apply SMOTE to the imbalance Dataset where our baseline is 60%
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12, ratio = .65)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

# %% Calculates Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(x_train_res, y_train_res)
clf_rf.score(test_features, test_target)

# %% Predicts the Y value using Random Forest
from sklearn import metrics
y_pred_rf = clf_rf.predict(test_features)
##a = metrics.accuracy_score(test_target, y_pred_rf)
##a

# %% Random Forest: Results 
from sklearn import metrics
print("Random Forest Results: ")
print("Accuracy:",metrics.accuracy_score(test_target, y_pred_rf))
print("Precision:",metrics.precision_score(test_target, y_pred_rf))
print("Recall:",metrics.recall_score(test_target, y_pred_rf))

# %%
#recall_score(test_target, clf_rf.predict(test_features))

# %% Random Forest: Confusion Matrix
import numpy as np
import seaborn as sb
cm_rf = metrics.confusion_matrix(test_target, y_pred_rf)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# Creates Confusion Matrix heatmap
sb.heatmap(cm_rf, annot=True, cmap="Blues_r" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#%% Random Forest: ROC/AUC Curve
y_pred_proba_clf = clf_rf.predict_proba(test_features)[::,1]
fpr_clf, tpr_clf, _ = metrics.roc_curve(test_target,  y_pred_proba_clf)
auc_clf = metrics.roc_auc_score(test_target, y_pred_proba_clf)
plt.title('ROC/AUC Curve for Random Forest', y=1.1)
plt.plot(fpr_clf,tpr_clf,label="data 1, auc="+str(auc_clf))
plt.legend(loc=4)
plt.show()

# %%
dataframe = pd.DataFrame.from_records(x_train_res)
dataframe.head(10)

# %%
merge.head(10)

# %% Random Forest: Feature importance
feature_importances = pd.DataFrame(clf_rf.feature_importances_, index = dataframe.columns,columns=['importance']).sort_values('importance', ascending=False)
feature_importances

#%%
#clf_rf.fit(x_train_res,y_train_res)
#pp.pprint(classification_report(clf_rf.predict(x_val_res), y_val_res))

# %% Calculates Muli-Layer Perceptron 
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
mlp = MLPClassifier(hidden_layer_sizes=(80, 100), max_iter=10,alpha=1e-4, solver='lbfgs', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)
mlp.fit(x_train_res, y_train_res)
y_pred_MLP = mlp.predict(test_features)
#a = metrics.accuracy_score(test_target, y_pred_MLP)
#a

#%%Multi-Layer Perceptron: Confusion Matrix
import numpy as np
import seaborn as sb
from sklearn.neural_network import MLPClassifier
cm_MLP = metrics.confusion_matrix(test_target, y_pred_MLP)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sb.heatmap(cm_MLP, annot=True, cmap="Blues_r" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Multi-Layer Perceptron', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#%% Multi-Layer Perceptron: Results 
print("\n\n Multi-Layer Perceptron Results: ")
print("Accuracy:",metrics.accuracy_score(test_target, y_pred_MLP))
print("Precision:",metrics.precision_score(test_target, y_pred_MLP))
print("Recall:",metrics.recall_score(test_target, y_pred_MLP))

#%% Multi-Layer Perceptron: ROC/AUC Curve
y_pred_proba_mlp = mlp.predict_proba(test_features)[::,1]
fpr_mlp, tpr_mlp, _ = metrics.roc_curve(test_target,  y_pred_proba_mlp)
auc_mlp = metrics.roc_auc_score(test_target, y_pred_proba_mlp)
plt.title('ROC/AUC Curve for Multi-Layer Perceptron', y=1.1)
plt.plot(fpr_mlp,tpr_mlp,label="data 1, auc="+str(auc_mlp))
plt.legend(loc=4)
plt.show()

#%%
#recall_score(test_target, y_pred_MLP)

# %% Calculates the Logistic Regression
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(x_train_res, y_train_res)
y_pred_LogReg = LogReg.predict(test_features)

# %% Logistic Regression: Confusion Matrix
cm_LR = metrics.confusion_matrix(test_target, y_pred_LogReg)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sb.heatmap(cm_LR, annot=True, cmap="Blues_r" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Logistic Regression', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# %% Logistic Regression: Results
from sklearn import metrics
from sklearn.metrics import mean_squared_error
print("\n\n Logistic Regression Results: ")
print("Accuracy:",metrics.accuracy_score(test_target, y_pred_LogReg))
print("Precision:",metrics.precision_score(test_target, y_pred_LogReg))
print("Recall:",metrics.recall_score(test_target, y_pred_LogReg))

#%% Logistic Regression: ROC/AUC Curve
y_pred_proba = LogReg.predict_proba(test_features)[::,1]
fpr, tpr, _ = metrics.roc_curve(test_target,  y_pred_proba)
auc = metrics.roc_auc_score(test_target, y_pred_proba)
plt.title('ROC/AUC Curve for Logistic Regression', y=1.1)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
#plt.savefig('log.png', facecolor=plt.get_facecolor(), transparent=True)

#%% Comparing the Models by plotting accuracies on a bar graph
from sklearn.metrics import mean_squared_error
error_lr = metrics.accuracy_score(test_target, y_pred_LogReg)
error_rf = metrics.accuracy_score(test_target, y_pred_rf)
error_mlp =metrics.accuracy_score(test_target, y_pred_MLP)
f = plt.figure(figsize=(10,5))
plt.bar(range(3),[error_lr,error_rf,error_mlp])
plt.xlabel("Classifiers")
plt.ylabel("Prediction Accuracy of Models ")
plt.title('Comparing Prediction Models by Accuracy', y=1.1)
plt.xticks(range(3),['Linear Regression','Random Forest','Multi-layer Preceptron'])
plt.legend(loc=3)
plt.show()
#%%