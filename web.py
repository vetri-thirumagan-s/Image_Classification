import streamlit as st
from classification import classify
from PIL import Image

st.set_page_config(page_title = "Image Classification", page_icon = '🔮')

st.title("Image Classification for Object Recognition")

input_src = st.file_uploader("upload here", type=['png', 'jpeg', 'jpg'])

st.image(input_src)
img = classify(input_src)

st.image(Image('result.jpg'))