import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


icon_path = 'icon.png'
st.set_page_config(page_title='Superconductivity Prediction App', page_icon=icon_path)


# Заголовок и описание приложения
st.title('Superconductivity Prediction App')
st.write('Welcome to the Superconductivity Prediction App! Enter a value and click "Predict".')
with open('catboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler for inverse transformation
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Replace 'scaler.pkl' with the actual filename of your scaler


# Streamlit app
def main():


    # Create input fields for user to input feature values
    features = ['gmean_atomic_mass', 'std_atomic_mass', 'gmean_fie', 'entropy_fie', 'wtd_entropy_fie',
                'wtd_range_fie', 'wtd_std_fie', 'gmean_atomic_radius', 'wtd_range_atomic_radius', 'mean_Density',
                'gmean_Density', 'wtd_gmean_Density', 'wtd_entropy_Density', 'wtd_range_Density', 'std_Density',
                'wtd_mean_ElectronAffinity', 'gmean_ElectronAffinity', 'wtd_range_ElectronAffinity',
                'std_ElectronAffinity', 'gmean_FusionHeat', 'wtd_range_FusionHeat', 'std_FusionHeat',
                'mean_ThermalConductivity', 'gmean_ThermalConductivity', 'entropy_ThermalConductivity',
                'wtd_entropy_ThermalConductivity', 'wtd_range_ThermalConductivity', 'std_ThermalConductivity',
                'wtd_gmean_Valence', 'wtd_range_Valence', 'std_Valence']

    col1, col2, col3 = st.columns(3)

    # Increase spacing between elements in columns
    col_spacing = 80  # Adjust this value to control spacing
    with col1:
        st.write("")  # Empty space to increase spacing
    with col2:
        st.write("")
    with col3:
        st.write("")

    input_data = []
    for i, feature in enumerate(features):
        with col1 if i < 10 else col2 if i < 20 else col3:
            value = st.text_input(f"Enter {feature}", key=feature)
            input_data.append(value)

    # Increase font size for labels
    st.markdown("<style>label{font-size: 16px !important;}</style>", unsafe_allow_html=True)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        unscaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        st.success(f"Predicted Critical Temperature: {unscaled_prediction[0][0]:.2f} K")


if __name__ == '__main__':
    main()
# Загрузка фонового изображения из текущей папки
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://catherineasquithgallery.com/uploads/posts/2021-02/thumbs/1613514898_2-p-fon-dlya-prezentatsii-po-teme-nauka-3.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()