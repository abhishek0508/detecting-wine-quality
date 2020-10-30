import pandas as pd
import numpy as np
from sklearn.svm import SVC

class Submission():
    def __init__(self, train_data_path, test_data_path):
        self.train_data = pd.read_csv(train_data_path, header=None)
        self.test_data = pd.read_csv(test_data_path)

    def predict(self):
        # Split the training data into x and y
        X_train,y_train = self.train_data.iloc[:,:-1], self.train_data.iloc[:,-1]
        
        # Train the model
        classifier = SVC(gamma='auto')
        classifier.fit(X_train, y_train)
        
        # Predict on test set and save the prediction
        submission = classifier.predict(self.test_data)
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv',header=['quality'],index=False)

