import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


st.header('VND BANKNOTES CLASSIFICATION')
st.image('media\isle_of_dog.gif')

col1,col2,col3 = st.columns(3)

with col3:
    st.write('Upload file from computer')
    file_upload = st.file_uploader('Upload file', type=['jpeg','jpg','png'])
    st.image(file_upload)
with col2: 
    st.write('OR')
with col1:   
    st.write('Capture with your webcam')
    cap = cv2.VideoCapture(0)  # device 0
    run = st.checkbox('Show Webcam')
    capture_button = st.button('Capture')

    captured_image = np.array(None)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    while run:
        ret, frame = cap.read()        
        # Display Webcam
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB ) #Convert color
        frame = cv2.flip(frame, 1)
        FRAME_WINDOW.image(frame)

        if capture_button:      
            captured_image = frame
            break

    cap.release()
    if  captured_image.all() != None:
        st.write('Image is captured')

model = tf.keras.models.load_model('model\my_model_save_1.h5')
if st.image(file_upload) != None:
    prediction = model.predict(st.image(file_upload))
else:
    #Resize the Image according with your model
    captured_image = cv2.resize(224,224)
    #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
    img_array  = np.expand_dims(captured_image, axis=0)
    #Check the img_array here
    st.write(img_array)
    prediction = model.predict(img_array)

st.write('The image is:', prediction, ' VND')