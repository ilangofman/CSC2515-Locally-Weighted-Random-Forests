import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_data(file_path):
	data =  pd.read_csv(file_path, header=0)
	target = data['target']
	X = data.drop(columns=['target'])

	scaler = StandardScaler().fit(X)
	X_scaled = scaler.transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=6)

	return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()

	
def load_tweet_data(file_path):
    pass

def load_MNIST_data(train_file_path, test_file_path):
	train_data = np.genfromtxt(train_file_path, delimiter=',')
	train_X = train_data[:,1:]
	train_y = train_data[:, 0]

	test_data = np.genfromtxt(test_file_path, delimiter=',')
	test_X = test_data[:,1:]
	test_y = test_data[:, 0]

	scaler = StandardScaler().fit(train_X)
	train_X_scaled = scaler.transform(train_X)
	test_X_scaled = scaler.transform(test_X)

	return train_X_scaled, test_X_scaled, train_y, test_y

	