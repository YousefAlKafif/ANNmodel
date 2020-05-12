# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 20:39:40 2018

@author: youse
"""

# DATA PRE-PROCESSING :-
# Importing the libraries
import numpy as np #for array manipulation & management etc.
import pandas as pd #for dataset importing & management etc. 

# Importing the dataset :
dataset = pd.read_csv('Churn_Modelling.csv')
# Extracting the IVs and DV values, which are then used to create the training&test sets.
X = dataset.iloc[:, 3:13].values #IVs
y = dataset.iloc[:, 13].values #Assigning my DV values from the data set.

# Encoding categorical data i.e. transforming alphabetical categories into numerical values :
# Encoding the IVs - Countries[1] & Gender[2]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #Converting countries column into numerical values.
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #Converting genders column into numerical values. i.e. 0 & 1.

# Creating dummy variables for Countries column [1] to prevent the model from over-valueing a country, utilizing the OneHotEncoder() method. 
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()#Converting the country column into dummy variables using the O.H.E object previously initialized.

# Removing one DummyVariable to avoid DummyVariable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set w/ 20% of the data used as a test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling -- Using the StandardScaler() module from the sklearn.preprocessing package.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) # sc is scaling on same fit basis as x_train


#----------------------------------------------------------------------------------
# BUILDING THE ANN :-
# Importing Keras packages : 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # The overfitting countermeasure
from tensorflow.contrib.keras.api.keras.callbacks import Callback

class LossHistory(Callback): # Creating a loss history class to log the models history.
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
     
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f} \n"\
            .format(str(self.epoch_id), logs.get('acc'))
        self.epoch_id += 1
         
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'

#Initializing LossHistory class object : 
history = LossHistory()

# Initializing the ANN - defining it as a sequence of layers :
classifier = Sequential()

# Adding the input layer and first hidden layer :
classifier.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform', input_shape=(11,)))
classifier.add(Dropout(rate = 0.1)) 
# Adding the second hidden layer :
classifier.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dropout(rate = 0.1))
# Adding our output layer :
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

# COMPILING THE ANN :
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# FITTING THE ANN/classifier TO OUR TRAINING SET - i.e. Training it - :
classifier.fit(X_train, y_train, batch_size = 50, epochs = 30, callbacks = [history])


# Saving model :
model_path = 'Saved/CustomerChurnANN.h5'
classifier.save(model_path)
print("Model saved to", model_path)
     
# Saving loss history to file : 
lossLog_path = 'Saved/loss_history.log'
myFile = open(lossLog_path, 'w+')
myFile.write(history.losses)
myFile.close()


# Making predictions :
y_pred = classifier.predict(X_test) # This gives us the probability of DV being 1. i.e. The customer leaving.
y_pred = (y_pred > 0.5) # Converting y_pred to a boolean value of True (above .5) or False. This needs to be done to ensure both y_pred and y_test are the same value type (boolean), and also if we want a 'yes/no' prediction.

# Making the Confusion Matrix to display correct and incorrect predictions :
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting a single new observation :
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


#----------------------------------------------------------------
# PARAMETER-TUNING THE ANN TO FIND OPTIMUM PARAMETER SETTINGS :- 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

#The parameters that will be tested :
parameters = {'batch_size': [25, 50],
              'epochs': [30, 100, 500],
              'optimizer': ['adam', 'rmsprop']}

# Intitializing the GridSearchCV with the correct configurations :
grid_search = GridSearchCV(estimator = classifier,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10) # Using 10 k-folds.

# Testing the different combinations of parameters :
grid_search = grid_search.fit(X_train, y_train)

# Feedback that I can use to fine tune the parameters :
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


#-----------------------------------------------------------------
# EVALUATING THE ANN USING THE K-FOLD TECHNIQUE THROUGH KERAS' cross_val_score MODULE :-
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 50, epochs = 30)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()
""" These are all the  basics of ANNs """