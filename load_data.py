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
