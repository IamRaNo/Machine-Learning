import math
import pickle

import numpy as np
import streamlit as st

pipe = pickle.load(open('model.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))
st.title('Laptop Price Predictor')

company = st.selectbox('Company', data['Company'].unique())
typeName = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('Ram(In GB)', data['Ram'].unique())
os = st.selectbox('OS', data['OpSys'].unique())
weight = st.number_input('Weight')
fhd_display = st.selectbox('Full HD', ['Yes', 'No'])
ips_display = st.selectbox('IPS Panel', ['Yes', 'No'])
touchscreen_display = st.selectbox('Touch Screen', ['Yes', 'No'])
uhd_display = st.selectbox('4k Display', ['Yes', 'No'])
cpu_type = st.selectbox('CPU Type', data['CPU Type'].unique())
cpu_speed = st.number_input('CPU Speed')
prm_mem = st.selectbox('Primary Memory Type', data['Primary Memory Type'].unique())
prm_mem_size = st.selectbox('Primary Memory Size', data['Primary Memory Capacity'].unique())
sec_mem = st.selectbox('Secondary Memory Type', data['Secondary Memory Type'].unique())
sec_mem_size = st.selectbox('Secondary Memory Size', data['Secondary Memory Capacity'].unique())
screen_size = st.number_input('Screen Size In Inches')
width_resolution = st.number_input('Width Resolution')
height_resolution = st.number_input('Height Resolution')
gpu = st.selectbox('GPU', data['GPU Brand'].unique())
if st.button('Predict Price'):
    if fhd_display == 'Yes':
        fhd_display = 1
    else:
        fhd_display = 0
    if ips_display == 'Yes':
        ips_display = 1
    else:
        ips_display = 0
    if uhd_display == 'Yes':
        uhd_display = 1
    else:
        uhd_display = 0
    if touchscreen_display == 'Yes':
        touchscreen_display = 1
    else:
        touchscreen_display = 0

    ppi = round(math.sqrt((math.pow(width_resolution, 2) + math.pow(height_resolution, 2))) / screen_size, 2)

    query = np.array(
        [company, typeName, ram, os, weight, fhd_display, ips_display, touchscreen_display, uhd_display, cpu_type,
         cpu_speed, prm_mem,
         prm_mem_size, sec_mem, sec_mem_size, gpu, ppi])
    query = query.reshape(1, 17)
    st.title(round(math.exp(pipe.predict(query)), 2))
