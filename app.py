import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Load Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)

# -----------------------------
# Train Model
# -----------------------------
model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------
# UI
# -----------------------------
st.title("🌳 Decision Tree Classifier - Iris Dataset")

st.write("Enter flower measurements:")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)

    species = iris.target_names[prediction][0]

    st.success(f"Predicted Flower: {species}")
