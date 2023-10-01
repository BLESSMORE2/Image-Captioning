import streamlit as st
import cv2
import os
import datetime
import glob
import shutil
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textwrap import wrap
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from pathlib import Path
import openai

with st.sidebar:
        st.write("DONE BY :")
        st.write("BLESSMORE MAJONGWE R197347M")
# Function to delete existing frames
def delete_existing_frames(output_frame_dir):
    existing_frames = glob.glob(os.path.join(output_frame_dir, '*.jpg'))
    for frame_file in existing_frames:
        os.remove(frame_file)
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Specify the path to the tokenizer.pickle file relative to the working directory
tokenizer_path = os.path.join(current_directory, 'tokenizer.pickle')

# Check if the file exists at the specified path
#if os.path.exists(tokenizer_path):
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)
#else:
   # print("tokenizer.pickle file not found at the specified path:", tokenizer_path)


# Specify the path to the model.h5 file relative to the working directory
model_path = os.path.join(current_directory, 'model.h5')
caption_model = load_model('model.h5', compile=False)
openai.api_key = "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"



# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)
# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
        
def get_completion(prompt, model=llm_model):
    messages = [{"role": "system", "content": "You are a helpful assistant that generates video descriptions."},
                {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]
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
        if not os.path.exists('frames'):
            os.makedirs('frames')
        else:
            shutil.rmtree('frames')
            os.makedirs('frames')
        output_frame_dir = "frames/"
        #output_frame_dir = Path(__file__).resolve().parents[1] / 'frames/'


        # Frame skip factor
        frame_skip_factor = 20
       
        # Delete existing frames
        delete_existing_frames(output_frame_dir)
        # filename =inputpath+"/frame"+str(count)+".jpg"
        # Convert the video_file object to a file path
        #video_path = os.path.join(output_frame_dir, "uploaded_video.mp4")
        #video_path = os.path.join(str(output_frame_dir), "uploaded_video.mp4")
        video_path = output_frame_dir+"uploaded_video.mp4"
        

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

        # elif option == "Generate Video Description":
        #     # Generate text that describes the video from the captions
        #     video_description = "\n".join(video_captions.values())
        #     st.write("Video Description:")
        #     st.write(video_description)
                
        elif option == "Generate Video Description":
            # Generate text that describes the video from the captions
            video_description = "\n".join(video_captions.values())

            # account for deprecation of LLM model
            # Get the current date
            current_date = datetime.datetime.now().date()

            # Define the date after which the model should be set to "gpt-3.5-turbo"
            target_date = datetime.date(2024, 6, 12)

            # Set the model variable based on the current date
            if current_date > target_date:
                llm_model = "gpt-3.5-turbo"
            else:
                llm_model = "gpt-3.5-turbo-0301"

            # Define a prompt to generate a video description
            prompt = f"Generate a description of what is happening in the video:\n{video_description}\nDescription:"

            # Generate the video description using OpenAI's GPT-3
            generated_description = get_completion(prompt, model=llm_model)

            # Display the generated description
            st.write("Generated Video Description:")
            st.write(generated_description)

if __name__ == "__main__":
    main()
