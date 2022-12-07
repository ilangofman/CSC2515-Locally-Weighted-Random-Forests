import numpy as np
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from typing import *

def euclidean_distance(x_1:np.ndarray, x_2:np.ndarray) -> float:
    return np.linalg.norm(x_1 - x_2)

def _mean(dataset: np.ndarray) -> np.ndarray:
    return np.mean(dataset, axis = 0)

class LocallyWeightedRandomForest(BaseEstimator,ClassifierMixin):


    def __init__(self, 
                 n_estimators:int=100, 
                 criterion:str="gini", 
                 max_depth: Union[int,None] = None, 
                 max_samples: Union[float,None] = None):

        '''
        Constructor for the model class
        Input: 
        n_estimators - number of estimators in the ensemble
        criterion - splitting criteria when training the individual trees
        max_depth - the max depth for each individual tree
        max_samples - the portion of the dataset subsampled for each tree. 
        '''

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

        self.max_samples = max_samples if max_samples else 1.0


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
        

        for _ in range(self.n_estimators):
            # First we sub-sample the dataset
            sampled_X, sampled_y = resample(X,y,n_samples=samples_to_draw,replace=sample_replace)

            _decision_tree = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion)
            _decision_tree.fit(sampled_X, sampled_y)
            self.estimators.append(_decision_tree)
            # we could probably just record the indexes that we sampled might be more
            # efficient if we have many estimators
            self.estimator_datasets[_decision_tree] = sampled_X, sampled_y
        

    def predict(self, test_X:np.ndarray, temperature:float = 1.0, distance_function:Callable = euclidean_distance, aggregation_function:Callable = _mean):
        '''
        Calculate the predictions given the distance function and the temperature value for 
        aggregating the distance values

        Input: test_X - the data to calculate the predictions with 
            distance_function - a function that takes in two parameters 
                    * point - a single point to predict on
                    * X - the dataset used to train the classifier
                    This function is meant to allow for a flexible calculation of distances which will get aggregated afterwards
                temperature - input to the distance softmax calculation

        Output: predictions numpy array 
        '''

        predictions = np.zeros(test_X.shape[0])

        for i, test_point in enumerate(test_X):
            estimator_predictions = {}
            estimator_distances = np.zeros(self.n_estimators)

            for j, _estimator in enumerate(self.estimators):
                sampled_dataset = self.estimator_datasets[_estimator]
                sampled_X = sampled_dataset[0]
                #TODO Check that I'm averaging each ferature and not each point
                sample_point = _mean(sampled_X)
                estimator_distances[j] = distance_function(test_point, sample_point)

            # Calculate the weights. Now all the weights should add to 1. 
            prediction_weights = self.calculate_weights(estimator_distances, temperature)

            # Predict the value using the estimators and the associated weights. 
            for j, _estimator in enumerate(self.estimators):
                # Make the prediction 
                est_prediction = _estimator.predict([test_point])[0]
                
                # If this class hasn't been predicted before, initialize the sum as 0. 
                if est_prediction not in estimator_predictions:
                    estimator_predictions[est_prediction] = 0

                # Add the weight of that prediction to the predicted class' running total
                estimator_predictions[est_prediction] += prediction_weights[j] 

            # alternative way to make predictions using the probability distribution of each model
            # res = np.zeros(self.estimators[0].n_classes_)
            # for j, _estimator in enumerate(self.estimators):
            #     res += _estimator.predict_proba([test_point]) * prediction_weights[j]
            
            # The final prediction will be the class with the largest sum of its weights
            # Get the argmax of the dictionary. I.e. key with the largest value
            predictions[i]  = max(estimator_predictions, key=estimator_predictions.get)

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
        total_den_sum = 0
        for distance in estimator_distances:
            total_den_sum += np.exp(-distance / (2 * temperature ** 2))

        for i, distance in enumerate(estimator_distances):
            weights[i] = np.exp(-distance / (2 * temperature ** 2)) / total_den_sum
        
        return weights

