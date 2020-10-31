import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class Submission():
    def __init__(self, train_data_path, test_data_path):
        self.train_data = pd.read_csv(train_data_path, header=None)
        self.test_data = pd.read_csv(test_data_path)

    def predict(self):
        # Split the training data into x and y
        X_train,y_train = self.train_data.iloc[:,:-1], self.train_data.iloc[:,-1]
        
        # Train the model
        # classifier = SVC(gamma='auto')
        # classifier.fit(X_train, y_train)
        classifier = RandomForestClassifier(random_state=2018, oob_score=True, min_samples_leaf=1, n_estimators=150, class_weight = {6:46, 5:29.2, 7:18, 8:3.5, 4:3.1, 3:0.4, 9:0.01})
        classifier.fit(X_train, y_train)

        # Predict on test set and save the prediction
        submission = classifier.predict(self.test_data)
        submission = pd.DataFrame(submission)
        submission.to_csv('submission.csv',header=['quality'],index=False)

