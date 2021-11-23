import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

menu = ['Upload a file', 'Capture with your webcam']

choice = st.sidebar.selectbox('Please choose:', menu)

model = tf.keras.models.load_model('my_model_checkpoint_2.h5')
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

labels = {0:'1000',1:'10000',2:'100000',3:'2000',4:'20000',5:'200000',6:'5000',7:'50000',8:'500000'}
IMG_SIZE = 224

if choice == 'Upload a file':
    st.header('VND BANKNOTES CLASSIFICATION')
    st.image('VNO-Bank Notes and Change.jpg')
    file_upload = st.file_uploader('Upload file', type=['jpeg','jpg','png'])
    st.image(file_upload)
    if file_upload != None:     
            image_np = np.asarray(bytearray(file_upload.read()), dtype = np.uint8)    
            img = cv2.imdecode(image_np,1)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            img_array  = np.expand_dims(img, axis=0)
            prediction = model.predict(img_array)
            pred_indices = np.argmax(prediction, axis = 1)

    else: st.write('No file uploaded')
    st.write('The image is:', labels[int(pred_indices)], ' VND')

if choice == 'Capture with your webcam':   
    st.header('VND BANKNOTES CLASSIFICATION')
    st.image('VNO-Bank Notes and Change.jpg')
    cap = cv2.VideoCapture(0)  # device 0
    run = st.button('Refresh')
    capture_button = st.button('Capture')

    captured_image = np.array(None)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    while True:
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
        #Resize the Image according with your model
        captured_image = cv2.resize(captured_image,(IMG_SIZE,IMG_SIZE))
        #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(captured_image, axis=0)
        #Check the img_array here
        # st.write(img_array)
        prediction = model.predict(img_array)
        pred_indices = np.argmax(prediction, axis = 1)
        st.write('The image is:', labels[int(pred_indices)], ' VND')