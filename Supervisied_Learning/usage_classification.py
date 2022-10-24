from xml.etree.ElementTree import PI
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

CURRENT_DIR = os.path.dirname(__file__)
path_file = os.path.join(CURRENT_DIR, 'data/diabetes.csv')

diabetes = pd.read_csv(path_file)
# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
label = 'Diabetic'

X, y = diabetes[features].values,  diabetes[label].values
for n in range(0, 4):
    print("Patient", str(n+1), "\n  Features:",list(X[n]), "\n  Label:", y[n])


features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
for col in features:
    diabetes.boxplot(column=col, by='Diabetic', figsize=(6,6))
    plt.title(col)
plt.show()


# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))


# run training section LogisticRegression with some hyperparameter
# Set regularization rate
reg = 0.01
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train,y_train)

# test the model with predict with X_test
prediction = model.predict(X_test)
print('Predicted labels: ', prediction)
print('Actual labels:    ' ,y_test)

print("Accuracy : ", accuracy_score(y_test, prediction))

from sklearn. metrics import classification_report
# use confusion matrix precision and recall with below explanation
'''
Precision: Of the predictions the model made for this class,
    what proportion were correct?
Recall: Out of all of the instances of this class in the test dataset, 
    how many did the model identify?
F1-Score: An average metric that takes both precision and recall into account.
Support: How many instances of this class are there in the test dataset?

'''
print(classification_report(y_test, prediction))

# for overall calculation precision and recall 
from  sklearn. metrics  import precision_score, recall_score

'''
Because this is a binary classification problem,
the 1 class is considered positive and its precision 
and recall are particularly interesting - these in effect answer the questions:

Of all the patients the model predicted are diabetic, how many are actually diabetic?
Of all the patients that are actually diabetic, how many did the model identify?
'''

print("Overall Precision:",precision_score(y_test, prediction))
print("Overall Recall:",recall_score(y_test, prediction))


from sklearn.metrics import confusion_matrix

'''
The precision and recall metrics are derived from four possible prediction outcomes:

True Positives: The predicted label and the actual label are both 1.
False Positives: The predicted label is 1, but the actual label is 0.
False Negatives: The predicted label is 0, but the actual label is 1.
True Negatives: The predicted label and the actual label are both 0.

 the shape of this matrix is    TN|FP
                                FN|TP

'''
# Print the confusion matrix
cm = confusion_matrix(y_test, prediction)
print (cm)

# you can see the probability of each pair between two class
y_scores = model.predict_proba(X_test)
print(y_scores)

'''
The decision to score a prediction as a 1 or a 0 depends on the threshold to 
which the predicted probabilities are compared. If we were to change the threshold, 
it would affect the predictions; and therefore change the metrics in the confusion 
matrix. A common way to evaluate a classifier is to examine the true positive rate 
(which is another name for recall) and the false positive rate for a range of possible 
thresholds. These rates are then plotted against all possible thresholds to form a 
chart known as a received operator characteristic (ROC) chart, like this:

'''

from sklearn.metrics import roc_curve
# calculte the ROC curve
fpr, tpr, thresholds =  roc_curve(y_test, y_scores[:,1])

# plot ROC Curvve
fig = plt.figure(figsize=(6, 6))
# plot diagonal  50% line
plt.plot([0, 1], [1,0], 'k--')
# Plot the FPR and TPR achieved by our model
# ROC is rate FP to TP
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

'''
The ROC chart shows the curve of the true and false positive rates
for different threshold values between 0 and 1. A perfect classifier
would have a curve that goes straight up the left side and straight 
across the top. 

The area under the curve (AUC) is a value between 0 and 1 that quantifies 
the overall performance of the model. The closer to 1 this value is, the 
better the model. Once again, scikit-Learn includes a function to calculate 
this metric.

'''

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))

# perform pipline for data preprocessing in Scaling_numeric featuers and 
# one_hot method for none binary featuers

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Define preprocessing for numeric columns (normalize them so they're on the same scale)
numeric_features = [0,1,2,3,4,5,6]
numeric_transfomer =  Pipeline(steps=[('scaler', StandardScaler())])


# Define preprocessing for categorical features (encode the Age column)
categorical_features = [7]
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])


# Combine preprocessing steps
preprocessor =  ColumnTransformer(
    transformers=[
        ('num', numeric_transfomer, numeric_features)
        ('cat', categorical_transformer, categorical_features)])

# Create preprocessing and training pipeline
pipline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logregressor', LogisticRegression(C=1/reg, solver="liblinear"))])

# fit the pipeline to train a logistic regression model on the training set
model = pipline.fit(X_train, (y_train))
print (model)


# Get predictions from test data
predictions = model.predict(X_test)
y_scores = model.predict_proba(X_test)

# Get evaluation metrics
cm = confusion_matrix(y_test, predictions)
print ('Confusion Matrix:\n',cm, '\n')
print('Accuracy:', accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# try diffrent algorithms
from sklearn.ensemble import RandomForestClassifier

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('logregressor', RandomForestClassifier(n_estimators=100))])

# fit the pipeline to train a random forest model on the training set
model = pipeline.fit(X_train, (y_train))
print (model)

predictions = model.predict(X_test)
y_scores = model.predict_proba(X_test)
cm = confusion_matrix(y_test, predictions)
print ('Confusion Matrix:\n',cm, '\n')
print('Accuracy:', accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))
auc = roc_auc_score(y_test,y_scores[:,1])
print('\nAUC: ' + str(auc))

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()



# at the end use model for infering
import joblib
import numpy as np
# Save the model as a pickle file
filename = './diabetes_model.pkl'
joblib.dump(model, filename)


# use the model
# Load the model from the file
model = joblib.load(filename)

# predict on a new sample
# The model accepts an array of feature arrays (so you can predict the classes of multiple patients in a single call)
# We'll create an array with a single array of features, representing one patient
X_new = np.array([[2,180,74,24,21,23.9091702,1.488172308,22]])
print ('New sample: {}'.format(list(X_new[0])))

# Get a prediction
pred = model.predict(X_new)

# The model returns an array of predictions - one for each set of features submitted
# In our case, we only submitted one patient, so our prediction is the first one in the resulting array.
print('Predicted class is {}'.format(pred[0]))