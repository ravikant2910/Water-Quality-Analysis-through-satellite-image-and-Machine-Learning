'''
from CIE import RGBtoCIE
import joblib
import numpy as np
model = joblib.load('wavelength_model.pkl')
x = list(map(int,input('Enter RGB: ').split()))
y = RGBtoCIE(x)
x_array = np.array(x)  # Convert the list to a NumPy array
predicted_wavelength = model.predict([x_array])  # Wrap x_array in a list
print(f"Predicted Wavelength: {predicted_wavelength[0]} nm")
'''
import pandas as pd
import joblib
import numpy as np
from CIE import RGBtoCIE

# Load the model
model = joblib.load('wavelength_model.pkl')

# Read the data from the CSV file
df = pd.read_csv("E:\RBL-Water Project\WAVELENGTH.csv")

# Define a function to calculate the error
def calculate_error(row):
    x = [row['R'], row['G'], row['B']]
    y = RGBtoCIE(x)
    predicted_wavelength = model.predict([np.array(x)])
    outcome = predicted_wavelength[0]
    error = row['wavelength'] - outcome
    return outcome, error

# Apply the function to each row and create new columns
df['Outcome'], df['Error'] = zip(*df.apply(calculate_error, axis=1))

# Save the updated DataFrame back to the CSV file
df.to_csv('WAVELENGTH.csv', index=False)

print("Data updated and saved to WAVELENGTH.csv")
