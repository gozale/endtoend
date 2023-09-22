import numpy as np
import pickle

# Guardar el modelo
loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

# Definir los datos
input_data = (-122.45, 37.78, 52, 3975, 716, 1515, 691, 5.0156, 500001, "NEAR BAY")

# extrar los datos numericos
numerical_features = list(input_data[:8])  # Exclude the categorical feature

# convertir la categoria near bay a one-hot encoded vector
categorical_feature = input_data[9]
categories = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]  # List of possible categories
one_hot_encoded = [1 if category == categorical_feature else 0 for category in categories]

# variables adicionales
rooms_per_household = input_data[2] / input_data[6]
bedrooms_per_room = input_data[3] / input_data[2]
population_per_household = input_data[4] / input_data[6]

input_data_processed = numerical_features + one_hot_encoded + [rooms_per_household, bedrooms_per_room, population_per_household]

input_data_as_npa = np.asarray(input_data_processed).reshape(1, -1)

# hacer la prediccion
prediction = loaded_model.predict(input_data_as_npa)
print(prediction)
