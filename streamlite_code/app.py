import streamlit as st
import cv2
import os
import glob
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textwrap import wrap
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm

# Function to delete existing frames
def delete_existing_frames(output_frame_dir):
    existing_frames = glob.glob(os.path.join(output_frame_dir, '*.jpg'))
    for frame_file in existing_frames:
        os.remove(frame_file)

# Load the tokenizer from the saved file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the caption model
caption_model = load_model('model.h5')

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text

# Streamlit app
def main():
    st.title("Video Frame Captioning App")
    st.write("Upload a video to extract frames and generate captions.")

    # Video file uploader
    video_file = st.file_uploader("Upload a video file", type=["mp4"])

    if video_file is not None:
        # Output frame directory
        output_frame_dir = 'frames/'

        # Frame skip factor
        frame_skip_factor = 20

        # Delete existing frames
        delete_existing_frames(output_frame_dir)

        # Convert the video_file object to a file path
        video_path = os.path.join(output_frame_dir, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        # Create a capture object
        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Check if the frame should be saved based on the frame_skip_factor
            if frame_count % frame_skip_factor == 0:
                frame_filename = os.path.join(output_frame_dir, f'frame_{frame_count:04d}.jpg')
                cv2.imwrite(frame_filename, frame)

        cap.release()
        st.success("Frames extracted successfully!")
        option=None
        if option == None:
            st.markdown("<p style='color:red;'>Generating frames discription and text that describes the video......................</p>", unsafe_allow_html=True)


        # Load the DenseNet201 model
        base_model = DenseNet201(weights='imagenet')
        fe_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

        # Define the image size
        img_size = 224

        # Define the directory containing video frames
        frame_dir = 'frames/'  # Update this to the directory where your video frames are stored

        # Initialize a dictionary to store frame features
        frame_features = {}

        # Iterate through video frames and extract features
        for frame_filename in os.listdir(frame_dir):
            frame_path = os.path.join(frame_dir, frame_filename)

            # Load and preprocess the image
            img = cv2.imread(frame_path)
            # Check if img is None (failed to read the image)
            if img is None:
                continue  # Skip this frame and continue with the next one
            img = cv2.resize(img, (img_size, img_size))
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            # Extract features using DenseNet201
            features = fe_model.predict(img, verbose=0)

            # Store the features in the dictionary using the frame filename as the key
            frame_features[frame_filename] = features

        # Initialize a dictionary to store video captions
        video_captions = {}

        # Iterate through frame features and generate captions
        for frame_filename, features in frame_features.items():
            # Generate a caption for the frame using the provided caption_model
            caption = predict_caption(caption_model, frame_filename, tokenizer, 34, frame_features)

            # Store the caption in the dictionary using the frame filename as the key
            video_captions[frame_filename] = caption

        # Create a dropdown menu for user selection
        option = st.selectbox("Select an option", ["click to select option", "Generate Video Description","Display Frames",])

        if option == "Display Frames":
            # Display captions for each frame with original resolution
            for frame_filename, caption in video_captions.items():
                frame = cv2.imread(os.path.join(output_frame_dir, frame_filename))
                st.image(frame, caption=f"Frame: {frame_filename}\nCaption: {caption}", use_column_width=False)
        elif option == "Generate Video Description":
            # Generate text that describes the video from the captions
            video_description = "\n".join(video_captions.values())
            st.write("Video Description:")
            st.write(video_description)

if __name__ == "__main__":
    main()