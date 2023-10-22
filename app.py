import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
import pandas as pd

# Define the architecture of the neural network model
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the pre-trained model
def load_model(model_path, input_size):
    device = torch.device("cpu")  # Map the model to the CPU
    model = Net(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Load the pre-trained model
model_path = "pm10_predictor_model2.pth"
input_size = 10
loaded_model = load_model(model_path, input_size)

# Load the training data for fitting the scaler
training_data = pd.read_csv("data.csv")
X_train = training_data[['CO', 'NO2', 'SO2', 'Temperature', 'Humudity', 'Noise', 'O3', 'Uray', 'AQI', 'PM2_5']]
scaler = RobustScaler()
scaler.fit(X_train)

# Create a Streamlit app
st.title("PM10 Predictor App")
st.write("Enter the following data to predict PM10:")

# Input fields for user input
co = st.number_input("CO", value=0.0)
no2 = st.number_input("NO2", value=0.0)
so2 = st.number_input("SO2", value=0.0)
temperature = st.number_input("Temperature", value=0.0)
humidity = st.number_input("Humidity", value=0.0)
noise = st.number_input("Noise", value=0.0)
o3 = st.number_input("O3", value=0.0)
uray = st.number_input("Uray", value=0.0)
aqi = st.number_input("AQI", value=0.0)
pm2_5 = st.number_input("PM2.5", value=0.0)

# Make predictions when the user clicks a button
if st.button("Predict PM10"):
    input_data = [co, no2, so2, temperature, humidity, noise, o3, uray, aqi, pm2_5]
    scaled_data = torch.Tensor(scaler.transform([input_data]))
    predicted_pm10 = loaded_model(scaled_data).item()
    st.write(f"Predicted PM10: {predicted_pm10:.2f}")
