#Importing necessary libraries
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

#Load dataset
df = pd.read_csv('onlinefoods.csv')

data_cleaned = df.drop(columns=['Unnamed: 12'])
data_encoded = pd.get_dummies(data_cleaned, columns=[
    'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
    'Educational Qualifications', 'Feedback', 'Output'
])

#Plot distributions of age and family size
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data_encoded['Age'], kde=True)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
sns.histplot(data_encoded['Family size'], kde=True)
plt.title('Family Size Distribution')

plt.show()

#Plot correlation matrix
plt.figure(figsize=(15, 10))
correlation_matrix = data_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Display basic information about the dataset
print(df.info())
print(df.describe())

#Check for missing values
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#Handling missing values
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

#Encoding categorical variables
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']
numerical_features = ['Age', 'Family size', 'latitude', 'longitude']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

#Splitting the dataset into training and testing sets
X = df.drop('Output', axis=1)
y = df['Output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Applying preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plt

#Contoh data
#X_train, X_test, y_train, y_test

#Encode target labels with value between 0 and n_classes-1
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#Initialize Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)

#Train the model
logreg_model.fit(X_train, y_train_encoded)

#Predict on test set
y_pred_encoded = logreg_model.predict(X_test)

#Calculate evaluation metrics
mae = mean_absolute_error(y_test_encoded, y_pred_encoded)
mse = mean_squared_error(y_test_encoded, y_pred_encoded)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_encoded, y_pred_encoded)

#Print evaluation metrics
print('Logistic Regression Metrics =')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r2:.2f}\n')

#Define models
models = {
    'Logistic Regression' : logreg_model.fit(X_train, y_train_encoded),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

#Train and evaluate models using cross-validation
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = cv_scores
    print(f'{name} Cross-Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})')

#Initialize models
models = {
    '\n Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

#Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

#Plot model performance
plt.figure(figsize=(10, 5))
plt.boxplot(results.values(), labels=results.keys())
plt.title('Model Comparison')
plt.ylabel('Cross-Validation Accuracy')
plt.show()

#Train the best model
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

#Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Model = Random Forest \nRandom Forest Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

#Handle missing values
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

#Encode categorical variables
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']
numerical_features = ['Age', 'Family size', 'latitude', 'longitude']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

#Split the dataset
X = df.drop('Output', axis=1)
y = df['Output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

#Save model and preprocessor
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

import streamlit as st
import pandas as pd
import numpy as np

# oad model and preprocessor
model = joblib.load('random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

#Input form for user
st.title('Prediksi Output untuk Online Foods')

#Input features
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Prefer not to say'])
occupation = st.selectbox('Occupation', ['Employee', 'House wife', 'Self Employeed', 'Student'])
monthly_income = st.selectbox('Monthly Income', ['Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000', 'No Income'])
educational_qualifications = st.selectbox('Educational Qualifications', ['School', 'Graduate', 'Post Graduate', 'Ph.D', 'Uneducated'])
feedback = st.selectbox('Feedback', ['Positive', 'Neutral', 'Negative'])
age = st.number_input('Age', min_value=0)
family_size = st.number_input('Family Size', min_value=0)
latitude = st.number_input('Latitude')
longitude = st.number_input('Longitude')

#Create DataFrame from inputs
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

#Button to make prediction
if st.button('Predict'):
    try:
        # Apply preprocessing
        user_input_processed = preprocessor.transform(user_input)
        
        # Make prediction
        prediction = model.predict(user_input_processed)
        prediction_proba = model.predict_proba(user_input_processed)
        
        # Display prediction
        st.write('### Hasil Prediksi')
        st.write(f'Output Prediksi: {prediction[0]}')
        st.write(f'Probabilitas Prediksi: {prediction_proba[0]}')
    except ValueError as e:
        st.error(f"Error during preprocessing: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
