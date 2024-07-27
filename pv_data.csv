import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate the CSV file
data = {
    'irradiance': [800, 750, 820, 780, 770, 810, 760, 820, 730, 800],
    'temperature': [25, 27, 24, 26, 28, 23, 29, 25, 27, 26],
    'previous_power': [100, 90, 110, 95, 92, 105, 88, 112, 85, 100],
    'optimal_duty_cycle': [0.5, 0.45, 0.55, 0.47, 0.46, 0.53, 0.44, 0.56, 0.43, 0.5]
}

df = pd.DataFrame(data)
df.to_csv('pv_data.csv', index=False)

# Load and preprocess data
data = pd.read_csv('pv_data.csv')

X = data[['irradiance', 'temperature', 'previous_power']]
y = data['optimal_duty_cycle']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ANN model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Absolute Error: {mae}')
