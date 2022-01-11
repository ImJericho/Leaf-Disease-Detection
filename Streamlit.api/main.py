import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
MODEL = tf.keras.models.load_model("saved_model")
CLASS_NAMES = ["has Early Blight diesease","has Late Blight diesease", "Healthy"]


st.title("PATATO DISEASE CLASSIFICATION")
# st.markdown("<h1 style='text-align: center; color: gray;'><B>PATATO DISEASE CLASSIFICATION</B></h1>", unsafe_allow_html=True)
st.write("*..............................This Project was owned by* Vivek Patidar *all the licence belongs to him................................*")


# taking image as input
image_type = st.selectbox('select image type',['jpg','png','jfif'])
uploaded_file = st.file_uploader("Choose an image...", type=image_type)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=False)
    image = np.array(image)
    image = np.expand_dims(image, 0)
    st.write("Classifying............")
    st.write(".")
    st.write(".")
    
    predictions = MODEL.predict(image)
    prediction = np.argmax(predictions)
    confidence = np.max(predictions)
    
    
    # st.markdown("<h2 style='text-align: center; color: gray;'><B>PATATO DISEASE CLASSIFICATION</B></h2>", unsafe_allow_html=True)
    st.write('This Plant is : %s' %(CLASS_NAMES[prediction]))
    st.write('Confidence = %f percentage' %(confidence*100))
