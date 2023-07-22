import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
loaded_model = load_model('model.h5')

# Class labels
class_labels = ['bike', 'car', 'truck', 'cart', 'bicycle']

# Image dimensions used for preprocessing (should be consistent with the model's input size)
image_width, image_height = 224, 224


# Function to preprocess the image
def preprocess_image(image):
    try:
        # Convert the image to RGB mode to ensure it has three color channels (R, G, B)
        image = image.convert('RGB')

        # Resize the image
        image = image.resize((image_width, image_height))

        # Convert the image to numpy array
        image = np.array(image)

        # Normalize the image
        image = image / 255.0

        return image
    except Exception as e:
        st.error("Error occurred during image preprocessing.")
        st.error(str(e))
        return None


# Function to predict the class of the image
def predict_image(image):
    image = preprocess_image(image)
    if image is not None:
        image = np.expand_dims(image, axis=0)
        prediction = loaded_model.predict(image)
        predicted_class = np.argmax(prediction)
        return class_labels[predicted_class]
    else:
        return None


def main():
    st.title('Image Classification App')
    st.write('Upload an image to classify it into one of the following classes: bike, car, truck, cart, bicycle')

    # Image upload widget
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Read the uploaded image
            image = Image.open(uploaded_file)

            # Check if the image is read successfully
            if image is not None:
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Predict the class of the uploaded image
                prediction = predict_image(image)
                if prediction is not None:
                    # Display the predicted class with a bigger font
                    st.write(f'Predicted Class: <span style="font-size:30px">{prediction}</span>',
                             unsafe_allow_html=True)
                else:
                    st.error("Error: Unable to predict the class.")
            else:
                st.error("Error: Unable to read the uploaded image.")
        except Exception as e:
            st.error("Error occurred while processing the image.")
            st.error(str(e))


if __name__ == '__main__':
    main()

