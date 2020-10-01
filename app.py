import streamlit as st
from fastai.vision import load_learner
from PIL import Image
from fastai.learner

path = ''
learn_inf = load_learner(path)

st.title("Are you touching your face?")
st.markdown("This is a web app to classify the person in the uploaded image as touching their face or not.")

uploaded_file = st.file_uploader("Upload an image of a person: ", type=['png', 'jpg', 'jpeg'])

def classify(img):
    with st.spinner('Analyzing...'):
        pred, pred_idx, probs = learn_inf.predict(img)
    return f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    # display_image = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    prediction = classify(img)
    st.write("Prediction: ", prediction)