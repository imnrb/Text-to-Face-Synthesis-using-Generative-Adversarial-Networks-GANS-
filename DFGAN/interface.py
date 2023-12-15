import streamlit as st
import requests
import base64
import io
from PIL import Image
import numpy as np


st.title("Image Generation")

caption = st.text_input("Enter the caption:")
#caption = f'''{caption}'''
print(caption)
with open(r"D:\BE_Major_Project\Projects\DFGAN/example_captions.txt", "w") as f:
    f.write(caption)
if st.button("Generate Image"):
    # send a request to the server with the caption
    response = requests.post("http://localhost:8000/predict", data={"caption": caption})

    # check if the request was successful
    if response.status_code == 200:
        # decode the base64 encoded image
        image = base64.b64decode(response.json()['image'])
        image = Image.open(io.BytesIO(image))
        execution_time = np.round(response.json()['time'],3)
        # display the generated image
        st.image(image, width=300)
        st.write("Execution time: ", execution_time, "seconds")
    else:
        st.error("Failed to generate image")



