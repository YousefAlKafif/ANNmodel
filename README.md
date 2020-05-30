# ANNmodel
This is an Artificial Neural Network Deep-Learning model I created to predict whether a customer will leave the bank using a customers geographical location, credit score, gender, age, tenure, balance, # of products purchased, credit card holder or not, active member or not, and estimated salary.  This is done to gain insight on the factors that affect a business' custom churn rate. I used a dataset that contained information on 10,000 mock customers which also included whether or not the customer had left the bank - which was used for validation in this supervised learning project. 

This was done using Keras' API that ran on TensorFlow. 

ANN Evaluation:-
  I managed to train the ANN to a final accuracy of 86.61%, I used the K-fold technique (using sklearn's cross_val_score() module) to evaluate the mean accuracy and variance of my model utilizing 10 folds. The mean accuracy is 85.8% and the variance is ~1%, indicating a fairly low bias.
  
 Methods I used:-
 
 Pre-processing the data :
  - I had to encode the categorical data (alphabetical categories i.e. words) into numerical values.
  - Creating 'Dummy Variables' with the OneHotEncoder module.
  - I used the StandardScaler() module to scale my test and training set.
 
 
 Improving the ANN :
  - Parameter-tuning using the GridSearchCV() module to test different combinations of parameters, which gave me feedback on what parameters would be optimal.
  - The 'Dropout' method in my ANNs hidden layers to prevent over-fitting.
  - Saving my ANNs loss history for records purposes.
  
  Please view my 'ANNcomprehensionNotes' for a step-by-step analysis and breakdown of my work and how I did it. 
  https://github.com/YousefAlKafif/ANNonKeras/blob/master/ANNcomprehensionNotes.pdf
  
