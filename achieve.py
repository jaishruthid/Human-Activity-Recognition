import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)

def main():
    st.title("Activity Detection")

    # Load the pre-trained model
    model = load_model("keras_model.h5", compile=False)

    # Load class names
    class_names = open("labels.txt", "r").readlines()

    # Upload and process image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Display results
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.subheader("Prediction")
        st.write(class_name[2:])
        st.write("Confidence Score:", confidence_score)

if __name__ == "__main__":
    main()
