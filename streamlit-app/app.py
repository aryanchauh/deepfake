import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import preprocess_input
from mtcnn import MTCNN
import time
import pandas as pd
import matplotlib.pyplot as plt
import base64
import tempfile
# TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parameters
TIME_STEPS = 30  # Frames per video
HEIGHT, WIDTH = 299, 299

# Model builder
def build_model(lstm_hidden_size=256, num_classes=2, dropout_rate=0.5):
    with tf.keras.backend.name_scope('model'):  # Add explicit name scope
        inputs = layers.Input(shape=(TIME_STEPS, HEIGHT, WIDTH, 3))
        base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
        base_model.trainable = False  # Freeze the base model
        
        x = layers.TimeDistributed(base_model)(inputs)
        x = layers.LSTM(lstm_hidden_size, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model

# Load model
model_path = r'D:\Pro-jects\ESE major project\COMBINED_best_Phase1.keras'
model = build_model()
model.load_weights(model_path)

def preprocess_image(image):
    """
    Preprocess image for model input
    
    Args:
        image (PIL.Image or numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize to expected input size
    image = cv2.resize(image, (WIDTH, HEIGHT))
    
    # Preprocess for Xception model
    image = preprocess_input(image)
    
    return image

def extract_faces_from_video(video_path, start_time=0, duration=2, num_frames=TIME_STEPS):
    """
    Extract faces from a specific time window in the video
    
    Args:
        video_path (str): Path to the video file
        start_time (float): Start time in seconds
        duration (float): Duration in seconds
        num_frames (int): Number of frames to extract
    
    Returns:
        tuple: (video_array, frames) or (None, None) if no faces detected
    """
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps
    
    # Check if the requested window is valid
    if start_time >= total_duration:
        return None, None
    
    # Calculate frame indices to sample within the window
    start_frame = int(start_time * fps)
    end_frame = min(int((start_time + duration) * fps), frame_count)
    
    # Calculate frames to sample
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    
    frames = []
    
    for idx in range(frame_count):
        success, frame = cap.read()
        if not success:
            break
        
        # Check if this frame should be processed
        if idx in frame_indices:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = detector.detect_faces(frame_rgb)
            
            if detections:
                # Get the first detected face
                x, y, width, height = detections[0]['box']
                x, y = max(0, x), max(0, y)
                x2, y2 = x + width, y + height
                
                # Extract face
                face = frame_rgb[y:y2, x:x2]
                
                # Convert to PIL Image and preprocess
                try:
                    face_image = Image.fromarray(face)
                    processed_face = preprocess_image(face_image)
                    frames.append(processed_face)
                except Exception as e:
                    # If face processing fails, use a zero array
                    frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))
            else:
                # If no face detected, use a zero array
                frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))
            
            # Stop if we have collected enough frames
            if len(frames) == num_frames:
                break
    
    cap.release()
    
    # If not enough frames were found, pad with the last frame or zeros
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])  # Pad with the last frame
        else:
            frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))
    
    # Convert to numpy array and expand dimensions
    video_array = np.expand_dims(np.array(frames), axis=0)
    
    return video_array, frames

def get_video_details(video_path):
    """
    Get video duration and dimensions
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    cap.release()
    return duration, width, height, fps

def make_prediction(video_path, start_time):
    """
    Make prediction on the selected video window
    
    Args:
        video_path: Path to the video file
        start_time: Start time in seconds for the 2-second window
    
    Returns:
        tuple: (predicted_class, probabilities, frames) or (None, None, None) if error
    """
    try:
        # Extract faces and video array from the specified time window
        video_array, frames = extract_faces_from_video(video_path, start_time=start_time, duration=2)
        
        # Validate the video array
        if video_array is None or video_array.shape[1] != TIME_STEPS:
            st.error("Unable to process video segment. Please ensure the selected portion contains clear, visible faces.")
            return None, None, None
        
        # Make prediction
        predictions = model.predict(video_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        probabilities = predictions[0]
        
        return predicted_class, probabilities, frames
    
    except Exception as e:
        st.error(f"An error occurred while processing the video: {str(e)}")
        return None, None, None

def generate_thumbnail(video_path, timestamp):
    """Generate a thumbnail at a specific timestamp"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()
    cap.release()
    
    if success:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        return None

# Streamlit UI
st.set_page_config(page_title="Not Ur Face", layout="wide")
st.markdown("<style>h1{font-size: 45px !important;}</style>", unsafe_allow_html=True)

# Create two columns for header and main content
header_col1, header_col2 = st.columns([1, 1])

def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to the uploaded image
image_path = "Image2.png"  # Ensure this is the correct path to your saved image

# Convert image to Base64
try:
    image_base64 = get_base64_image(image_path)
except:
    image_base64 = ""  # Default empty if image not found

# Header Section with Image
with header_col1:
    try:
        image = Image.open("Image2.png")
        desired_height = 300  # Reduced height
        aspect_ratio = image.width / image.height
        new_width = int(desired_height * aspect_ratio)
        resized_image = image.resize((new_width, desired_height))
        # st.image(resized_image, use_container_width=True)
    except:
        pass  # Skip if image not found

# Title and Description
with header_col2:
    st.markdown(
    """
    <style>
    .header-container {
        position: relative;
        text-align: center;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .header-image {
        width: 100%;
        height: 300px;
        object-fit: cover;
    }
    .header-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 50px;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# HTML content for the header
if image_base64:
    st.markdown(
        f"""
        <div class="header-container">
            <img src="data:image/png;base64,{image_base64}" class="header-image" />
            <div class="header-text">NOT UR FACE: Video Analysis for Real & Synthetic Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Fallback if image is not available
    st.title("NOT UR FACE: Video Analysis for Real & Synthetic Detection")

# Sidebar
st.sidebar.title("How It Works")
st.sidebar.markdown(
    """
1. 📤 **Upload Video:** 
   - Choose a video file (mp4, mov, avi)
2. 🎯 **Select Time Window:**
   - Choose a starting point for the 2-second window from your video
3. 🔍 **Process Frames:** 
   - Detect and analyze faces
4. 🤖 **AI Analysis:** 
   - Predict 'Real' or 'Fake'
5. 📊 **Detailed Results:** 
   - View probabilities and insights

**Disclaimer:** The model is trained on FaceForensics++ and CelebDFV2 datasets so it works well on deepfake generation techniques used in these datasets. Model may not perform well for AI generated videos.
"""
)
st.sidebar.info(f"""Made by: Sarvansh Pachori✨         
                **GitHub:** sarvansh30""" )

# Upload video
st.subheader("🎥 Upload Your Video")
video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], label_visibility="collapsed")

if video_file is not None:
    # Save the uploaded video to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, f"temp_video.mp4")
    with open(temp_file, "wb") as f:
        f.write(video_file.read())
    
    # Get video details
    video_duration, video_width, video_height, fps = get_video_details(temp_file)
    
    # Create columns for video display and window selection
    video_col, selection_col = st.columns([3, 2])
    
    with video_col:
        st.subheader("Uploaded Video")
        st.video(temp_file)
    
    with selection_col:
        st.subheader("Select 2-Second Window")
        # Ensure max value is at least the video duration
        max_start_time = max(0, video_duration - 2)
        
        # Show slider for selecting start time
        start_time = st.slider(
            "Select starting point (seconds):",
            min_value=0.0,
            max_value=max_start_time,
            value=0.0,
            step=0.5
        )
        
        # Show thumbnail of selected starting point
        thumbnail = generate_thumbnail(temp_file, start_time)
        if thumbnail is not None:
            # Fixed: Changed use_column_width to use_container_width
            st.image(thumbnail, caption=f"Starting at {start_time:.1f}s", use_container_width=True)
        
        # Process button
        process_button = st.button("Process Selected Window", key="process_window")
    
    # Process the selected window when the button is clicked
    if process_button:
        st.subheader("Analysis of Selected Window")
        
        # Analysis columns
        results_col1, results_col2 = st.columns([1, 1])
        
        with results_col1:
            # Loading animation
            with st.spinner("🚀 Processing video window... Please wait!"):
                start_process_time = time.time()
                predicted_class, probabilities, frames = make_prediction(temp_file, start_time)
                end_process_time = time.time()
                processing_time = end_process_time - start_process_time
            
            if predicted_class is None:
                st.error("No faces detected in the selected window. Please select a different portion of the video.")
            else:
                # Display results
                if predicted_class == 0:
                    st.success("The selected video window is classified as **Real**!")
                else:
                    st.error("The selected video window is classified as **Fake**!")
                
                st.write(f"**Prediction Confidence:**")
                st.progress(int(probabilities[predicted_class] * 100))
        
        with results_col2:
            if predicted_class is not None:
                st.subheader("Class Probabilities")
                st.bar_chart({"Real": [probabilities[0]], "Fake": [probabilities[1]]})
        
        # Additional tabs for detailed results
        if predicted_class is not None:
            tab1, tab2 = st.tabs(["🖼️ Frame Previews", "⏱️ Processing Details"])
            
            with tab1:
                st.subheader("Frame Previews")
                st.write("Key frames analyzed during the process:")
                cols = st.columns(5)
                for i, frame in enumerate(frames[:10]):
                    frame = np.clip(frame, 0, 1)
                    frame = (frame * 255).astype(np.uint8)
                    with cols[i % 5]:
                        # Fixed: Changed use_column_width to use_container_width
                        st.image(frame, caption=f"Frame {i+1}", use_container_width=True)
            
            with tab2:
                st.subheader("Processing Details")
                st.write(f"**Time Window:** {start_time:.1f}s to {min(start_time + 2, video_duration):.1f}s")
                st.write(f"**Processing Time:** {processing_time:.2f} seconds")
                st.write(f"**Frames Analyzed:** {TIME_STEPS}")
                st.write(f"**Video FPS:** {fps:.2f}")
    
    # Clean up temp files when done
    try:
        os.remove(temp_file)
        os.rmdir(temp_dir)
    except:
        pass  # Ignore clean-up errors
else:
    # Display placeholder when no video is uploaded
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; 
                   height: 300px; border: 2px dashed #aaa; border-radius: 5px;">
            <div style="text-align: center;">
                <h3>Upload a video to get started</h3>
                <p>Supported formats: MP4, MOV, AVI</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )