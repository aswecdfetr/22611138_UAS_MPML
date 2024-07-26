import pickle 
import streamlit as st
import pandas as pd

#Membaca model
onlinefood_model = pickle.load(open('onlinefood_model.sav', 'rb'))

import pickle

# Simpan model dan preprocessor
filename = 'onlinefood_model.sav'
pickle.dump(best_model, open(onlinefood_model.sav, 'wb'))
preprocessor_filename = 'preprocessor.pkl'
pickle.dump(preprocessor, open(preprocessor.pkl, 'wb'))

# Muat model dan preprocessor
best_model = pickle.load(open(onlinefood_model.sav, 'rb'))
preprocessor = pickle.load(open(prepocessor.pkl, 'rb'))

# Aplikasi Streamlit
st.title('Prediksi Pembelian Makanan Online')

# Input fitur
gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
marital_status = st.selectbox('Status Perkawinan', ['Lajang', 'Menikah', 'Lebih baik tidak mengatakan'])
occupation = st.selectbox('Pekerjaan', ['Karyawan', 'Ibu Rumah Tangga', 'Wiraswasta', 'Pelajar'])
monthly_income = st.selectbox('Pendapatan Bulanan', ['Di bawah Rp.10000', '10001 hingga 25000', '25001 hingga 50000', 'Lebih dari 50000', 'Tidak ada pendapatan'])
educational_qualifications = st.selectbox('Kualifikasi Pendidikan', ['Sekolah', 'Sarjana', 'Pasca Sarjana', 'Ph.D', 'Tidak Berpendidikan'])
feedback = st.selectbox('Umpan Balik', ['Positif', 'Netral', 'Negatif'])
age = st.number_input('Usia', min_value=0)
family_size = st.number_input('Ukuran Keluarga', min_value=0)
latitude = st.number_input('Lintang')
longitude = st.number_input('Bujur')

# Buat DataFrame dari input
user_input = pd.DataFrame({
    'Gender': [gender],
    'Marital Status': [marital_status],
    'Occupation': [occupation],
    'Monthly Income': [monthly_income],
    'Educational Qualifications': [educational_qualifications],
    'Feedback': [feedback],
    'Age': [age],
    'Family size': [family_size],
    'latitude': [latitude],
    'longitude': [longitude]
})

# Tombol untuk membuat prediksi
if st.button('Prediksi'):
    try:
        # Terapkan preprocessing
        user_input_encoded = preprocessor.transform(user_input)

        # Buat prediksi
        prediction = best_model.predict(user_input_encoded)
        prediction_proba = best_model.predict_proba(user_input_encoded)

        # Tampilkan hasil prediksi
        st.write('### Hasil Prediksi')
        st.write(f'Prediksi Output: {prediction[0]}')
        st.write(f'Probabilitas Prediksi: {prediction_proba[0]}')
    except ValueError as e:
        st.error(f"Terjadi kesalahan selama preprocessing: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")