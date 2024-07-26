import pickle
import streamlit as st
import numpy as np

# Membaca model
weather_model = pickle.load(open('onlinefood_model.sav','rb'))

# Judul web
st.title('Prediksi Cuaca')

# Input data dengan contoh angka valid untuk pengujian
Precipitation = st.text_input('Segala bentuk air yang jatuh ke permukaan tanah')
temp_max = st.text_input('Suhu Maksimum')
temp_min = st.text_input('Suhu Minimum')
wind = st.text_input('Kecepatan Angin')

Prediksi_Cuaca = ''

# Membuat tombol untuk prediksi
if st.button('Prediksi'):
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[float(Precipitation), float(temp_max), float(temp_min), 
                            float(wind)]])
        # Lakukan prediksi
        cuaca_prediksi = weather_model.predict(inputs)
        
        if cuaca_prediksi[0] == 0:
            cuaca_prediksi = 'Drizzle'
            st.success(cuaca_prediksi)
        if cuaca_prediksi[0] == 1:
            cuaca_prediksi = 'Fog'
            st.success(cuaca_prediksi)
        if cuaca_prediksi[0] == 2:
            cuaca_prediksi = 'Rain'
            st.success(cuaca_prediksi)
        if cuaca_prediksi[0] == 3:
            cuaca_prediksi = 'Snow'
            st.success(cuaca_prediksi)
        if cuaca_prediksi[0] == 4:
            cuaca_prediksi = 'Sun'
            st.success(cuaca_prediksi)
    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")