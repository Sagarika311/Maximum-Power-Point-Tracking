import pandas as pd

data = {
    'irradiance': [800, 750, 820, 780, 770, 810, 760, 820, 730, 800],
    'temperature': [25, 27, 24, 26, 28, 23, 29, 25, 27, 26],
    'previous_power': [100, 90, 110, 95, 92, 105, 88, 112, 85, 100],
    'optimal_duty_cycle': [0.5, 0.45, 0.55, 0.47, 0.46, 0.53, 0.44, 0.56, 0.43, 0.5]
}

df = pd.DataFrame(data)
df.to_csv('pv_data.csv', index=False)
