import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess data
data = pd.read_csv('pv_data.csv')  # Assuming a CSV file with relevant data
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

def mppt_using_ann(irradiance, temperature, previous_power):
    input_data = np.array([[irradiance, temperature, previous_power]])
    optimal_duty_cycle = model.predict(input_data)
    return optimal_duty_cycle[0][0]

# Example values
current_irradiance = 800  
current_temperature = 25
previous_power = 100
optimal_duty = mppt_using_ann(current_irradiance, current_temperature, previous_power)
print(f'Optimal Duty Cycle: {optimal_duty}')
