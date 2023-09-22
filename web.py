import numpy as np
import pickle
import streamlit as st
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

def house_prediction(input_data):
    #input_data = (-122.45, 37.78, 52, 3975, 716, 1515, 691, 5.0156, 500001, "NEAR BAY")

    # extrar los datos numericos
    numerical_features = list(input_data[:8])  # Exclude the categorical feature

    # convertir la categoria near bay a one-hot encoded vector
    categorical_feature = input_data[9]
    categories = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]  # List of possible categories
    one_hot_encoded = [1 if category == categorical_feature else 0 for category in categories]

    ncuartos = int(input_data[3])
    nrecamaras = int(input_data[4])
    population = int(input_data[5])
    households = int(input_data[6])

    # variables adicionales
    rooms_per_household = ncuartos / households
    bedrooms_per_room = nrecamaras / ncuartos
    population_per_household = population / households

    input_data_processed = numerical_features + one_hot_encoded + [rooms_per_household, bedrooms_per_room, population_per_household]

    input_data_as_npa = np.asarray(input_data_processed).reshape(1, -1)

    # hacer la prediccion
    prediction = loaded_model.predict(input_data_as_npa)
    return prediction[0]


def main():

    #titulo
    st.title('Predicción de precios de casa')

    #input de datos
    longitud = st.slider("Longitude",-180,+180)
    latitud = st.slider("Latitude",-90,+90)
    edad_media = st.slider("Housing Median Age",0,100)
    ncuartos = st.text_input('Total Rooms')
    nrecamaras = st.text_input('Total Bedrooms')
    population = st.text_input('Population')
    households = st.text_input('Households')
    mincome = st.text_input('Median Income')
    mhousevalue = st.text_input('Median House Value')
    ptb = st.radio("Elige una", ["<1H OCEAN","INLAND","ISLAND","NEAR BAY","NEAR OCEAN"])

    #codigo para prediccion
    precio = 0


    if st.button('Cotización de la casa'):
        precio = house_prediction([longitud, latitud, edad_media, ncuartos, nrecamaras,population, households, mincome, mhousevalue, ptb])
        st.success(precio)



if __name__ == '__main__':
    main()