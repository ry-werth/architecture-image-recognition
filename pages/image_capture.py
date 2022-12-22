import json
from io import BytesIO
from PIL import Image
import os
import torch
from torchvision import transforms
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Picture Capture", page_icon="")

MODEL_PATH = 'models/second_model_random_horizontal_random_vertical_no_random_crop.pt'
LABELS_PATH = 'src/labels.txt'

def load_model(model_path):
    #model = torch.load(model_path, map_location='cpu')

    device = torch.device('cpu')
    
    model = torch.jit.load(model_path, map_location=device)


    model.eval()
    return model

def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories

def predict(model, categories, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top3_prob, top3_catid = torch.topk(probabilities, 3)
    for i in range(top3_prob.size(0)):
        st.write(categories[top3_catid[i]], top3_prob[i].item())

def take_picture():

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer as a 3D uint8 tensor with PyTorch:
        bytes_data = img_file_buffer.getvalue()
        st.image(bytes_data)
        return Image.open(BytesIO(bytes_data))
        
    else:
        return None


def main():

    st.title('Custom model demo')
    model = load_model(MODEL_PATH)
    categories = load_labels(LABELS_PATH)

    live_image = take_picture()
    live_result = st.button('Run on captured image')
    if live_result:
        if live_image is not None:

            st.write('Calculating results...')
            predict(model, categories, live_image)
        
        else:
            st.write('Please Take a Photo')


if __name__ == '__main__':

    main()