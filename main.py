import cv2
import streamlit as st
import numpy as np
# from PIL import Image
import tensorflow as tf
import os


def predict_class(img):
    
    # class names
    MODEL_NUMBER = max([int(i) for i in os.listdir("Saved_Models")])
    CLASS_NAMES = []
    for i in os.listdir("Dataset"):
        CLASS_NAMES.append(i)

    #model building
    MODEL = tf.keras.models.load_model("Saved_Models/"+str(MODEL_NUMBER))

    img = np.array(img)
    img = np.expand_dims(img, 0)
    predictions = MODEL.predict(img)
    prediction = np.argmax(predictions)
    confidence = np.max(predictions)
    pred = CLASS_NAMES[prediction]
    return pred, confidence
    


# GUI
st.title("TOMATO LEAF DISEASE DETECTION APP")
# st.markdown("<h1 style='text-align: center; color: gray;'><B>PATATO DISEASE CLASSIFICATION</B></h1>", unsafe_allow_html=True)
st.write("*This Project was owned by* Vivek Patidar *all the licence belongs to him*")



img_file_buffer = st.camera_input("Take The Picture Of The Leaf")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # function call
    class_name , confidence = predict_class(image)
    st.subheader('Prediction : %s' %(class_name))
    st.subheader('Confidence : %f percentage' %(confidence*100))

    # st.image(image, caption='Uploaded Image.', use_column_width=True)