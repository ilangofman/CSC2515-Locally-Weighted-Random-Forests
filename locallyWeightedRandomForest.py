import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from typing import *
import random

class LocallyWeightedRandomForest(BaseEstimator, ClassifierMixin):

    def __init__(self, 
                 n_estimators:int=100, 
                 criterion:str="gini", 
                 max_depth: Union[int,None] = None, 
                 max_samples: Union[float,None] = None,
                 temp: float = 1,
                 distance_function:Callable = lambda a,b: 1,
                 distance_aggregation_function:Callable  = lambda point,dataset,distance_func: 1):

        '''
        Constructor for the model class
        Input: 
        n_estimators - number of estimators in the ensemble
        criterion - splitting criteria when training the individual trees
        max_depth - the max depth for each individual tree
        max_samples - the portion of the dataset subsampled for each tree. 
                       distance_function - a function that takes in two points and returns the distances between them
        temperature - input to the distance softmax calculation
        distance_aggregation_function - a function that takes in three parameters 
            * point - a single point to predict on
            * dataset - the dataset used to train the classifier
            * distance_func - Which will be the distance function passed in. 
            This function determines how to aggragate the distances between the test point and the dataset. It 
            aims to provide a flexible approach to calculating the distance in different ways. 
        '''

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

        self.max_samples = max_samples if max_samples else 1.0
        self.temp = temp
        self.distance_function = distance_function
        self.distance_aggregation_function = distance_aggregation_function

    def get_params(self, deep=True):
        return {
            "n_estimators" : self.n_estimators,
            "criterion" : self.criterion,
            "max_depth" : self.max_depth,
            "max_samples" : self.max_samples,
            "temp" : self.temp,
            "distance_function" : self.distance_function,
            "distance_aggregation_function" : self.distance_aggregation_function
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


    def fit(self, X:np.ndarray, y:np.ndarray, sample_replace:bool = True):

        '''
        Fit the dataset to an ensemble of decision trees. Each decision tree
        is trained on a subsample of the dataset determined by the "max_samples" values
        of the model. 

        Input: X - the dataset feature values
            y - the target values corresponding to the dataset
        '''
        self.estimators : List[ClassifierMixin] = []
        self.estimator_datasets= {}

        total_samples = y.shape[0]
        samples_to_draw = int(total_samples * self.max_samples)
        self.train_X = X
        self.train_y = y

        for _ in range(self.n_estimators):
            # First we sub-sample the dataset
            # sampled_X, sampled_y = resample(X,y,n_samples=samples_to_draw,replace=sample_replace)
            sampled_index  = random.choices(range(0, len(X)), k=samples_to_draw)
            sampled_X = X[sampled_index]
            sampled_y = y[sampled_index]

            _decision_tree = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion)
            _decision_tree.fit(sampled_X, sampled_y)
            self.estimators.append(_decision_tree)
            # we could probably just record the indexes that we sampled might be more
            # efficient if we have many estimators
            self.estimator_datasets[_decision_tree] = sampled_index
        

    def predict(self, 
                test_X:np.ndarray):
        '''
        Calculate the predictions given the distance function and the temperature value for 
        aggregating the distance values

        Input: test_X - the data to calculate the predictions with 
        Output: predictions numpy array 
        '''

        # predictions = np.zeros(test_X.shape[0])

        # for i, test_point in enumerate(test_X):
        #     estimator_predictions = {}
        #     estimator_distances = np.zeros(self.n_estimators)

            # for j, _estimator in enumerate(self.estimators):
            #     sampled_index = self.estimator_datasets[_estimator]
            #     sampled_X = self.train_X[sampled_index]
            #     estimator_distances[j] = self.distance_aggregation_function(test_point, sampled_X, self.distance_function)

        #     # Calculate the weights. Now all the weights should add to 1. 
        #     prediction_weights = self.calculate_weights(estimator_distances, self.temp)

        #     # Predict the value using the estimators and the associated weights. 
        #     for j, _estimator in enumerate(self.estimators):
        #         # Make the prediction 
        #         est_prediction = _estimator.predict([test_point])[0]
                
        #         # If this class hasn't been predicted before, initialize the sum as 0. 
        #         if est_prediction not in estimator_predictions:
        #             estimator_predictions[est_prediction] = 0

        #         # Add the weight of that prediction to the predicted class' running total
        #         estimator_predictions[est_prediction] += prediction_weights[j] 

        #     # alternative way to make predictions using the probability distribution of each model
        #     # res = np.zeros(self.estimators[0].n_classes_)
        #     # for j, _estimator in enumerate(self.estimators):
        #     #     res += _estimator.predict_proba([test_point]) * prediction_weights[j]
            
        #     # The final prediction will be the class with the largest sum of its weights
        #     # Get the argmax of the dictionary. I.e. key with the largest value
        #     predictions[i]  = max(estimator_predictions, key=estimator_predictions.get)
        return self.predict_proba(test_X).argmax(axis=1)
    
    def predict_proba(self, 
                test_X:np.ndarray):
        '''
        Calculate the prediction probability given the distance function and the temperature value for 
        aggregating the distance values
        Input: test_X - the data to calculate the predictions with 
               distance_function - a function that takes in two points and returns the distances between them
               temperature - input to the distance softmax calculation
               distance_aggregation_function - a function that takes in three parameters 
                    * point - a single point to predict on
                    * dataset - the dataset used to train the classifier
                    * distance_func - Which will be the distance function passed in. 
                    This function determines how to aggragate the distances between the test point and the dataset. It 
                    aims to provide a flexible approach to calculating the distance in different ways. 
        Output: predictions numpy array 
        '''

        predictions = np.zeros((test_X.shape[0], self.estimators[0].n_classes_))

        for i, test_point in enumerate(test_X):
            estimator_distances = np.zeros(self.n_estimators)
            current_res = np.zeros((self.n_estimators, self.estimators[0].n_classes_))
            for j, _estimator in enumerate(self.estimators):
                sampled_index = self.estimator_datasets[_estimator]
                sampled_X = self.train_X[sampled_index]
                estimator_distances[j] = self.distance_aggregation_function(test_point, sampled_X, self.distance_function)
                current_res[j,:] = _estimator.predict_proba([test_point])

            # Calculate the weights. Now all the weights should add to 1. 
            prediction_weights = self.calculate_weights(estimator_distances, self.temp)
            current_res = current_res * prediction_weights.reshape(-1,1)

            predictions[i,:] = current_res.sum(axis=0)

        return predictions
            
            
    def calculate_weights(self, estimator_distances:np.ndarray, temperature:float):
        '''
        Calculate the weights of the trees using the distances.
        The weights are the softmax output of the distances. 

        Input: - estimator_distances: list of the distances of the point to each tree in the ensemble
            - temperature - hyperparameter for the softmax function. 
        
        Output: List of the weight values, the sum should be equal to 1. 

        '''

        weights = np.zeros(self.n_estimators)
        # TODO Figure out, should it be distance or -1 * distance because closer distances should get a larger value?

        # Calculate Denominator values
        # total_den_sum = 0
        # for distance in estimator_distances:
        #     total_den_sum += np.exp(-distance / (2 * temperature ** 2))

        # for i, distance in enumerate(estimator_distances):
        #     weights[i] = np.exp(-distance / (2 * temperature ** 2)) / total_den_sum
        weights = np.exp(- estimator_distances / (2 * temperature ** 2))

        weights = weights / weights.sum()
        
        return weights

