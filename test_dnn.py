import slopestabilitytools
import slopestabilityML
import pandas as pd
import settings
settings.init()

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

test_results = slopestabilitytools.datamanagement.import_tests()
is_success = slopestabilitytools.folder_structure.create_folder_structure()

test_results_combined = pd.DataFrame()
test_training, test_prediction = slopestabilityML.split_dataset(test_results.keys(), 999)
for name in test_training:
    test_results_combined = test_results_combined.append(test_results[name])
test_results_combined = test_results_combined.reset_index()
test_results_combined = test_results_combined.drop(['index'], axis='columns')
x_train, y_train = slopestabilityML.preprocess_data(test_results_combined)

model = Sequential()
model.add(Dense(500, activation='relu', input_dim=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)