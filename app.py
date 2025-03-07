import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import platform
import time

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .stAlert {
        background-color: #f8f9fa;
        border: 1px solid #eee;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .metric-container {
        background-color: #f1f8e9;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Detection Settings</h2>", unsafe_allow_html=True)
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        'Confidence Threshold', 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Adjust to filter out low-confidence detections"
    )
    
    # Input source selection
    st.markdown("### Input Source")
    source_radio = st.radio(
        "Select input source:",
        ["Webcam", "Upload Video"]
    )
    
    # Display info about the app
    with st.expander("About this app"):
        st.markdown("""
        This app uses YOLOv8 for real-time object detection.
        
        - **YOLOv8**: State-of-the-art object detection model
        - **Supported objects**: 80 different classes including people, cars, animals, etc.
        - **Performance**: Speed depends on your hardware
        
        Created with Streamlit and YOLOv8.
        """)

# Main content
st.markdown("<h1 class='main-header'>Real-Time Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Detect objects in real-time using YOLOv8</p>", unsafe_allow_html=True)

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

# Status message while loading model
with st.spinner("Loading YOLOv8 model..."):
    model = load_model()
    st.success(f"Model loaded successfully! Ready to detect {len(model.names)} different objects.")

# Create two columns for metrics
col1, col2 = st.columns(2)
with col1:
    fps_display = st.empty()  # Placeholder for FPS display

with col2:
    detection_count = st.empty()  # Placeholder for detection count

# Main display area
stframe = st.empty()

# Check if running on macOS
is_macos = platform.system() == 'Darwin'

# Process based on selected source
if source_radio == "Webcam":
    try:
        # For macOS, try different camera indices
        if is_macos:
            camera_indices = [0, 1, 2]  # Try indices 0, 1, and 2 for macOS
            cap = None
            for idx in camera_indices:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    break
                cap.release()
        else:
            cap = cv2.VideoCapture(0)

        if not cap or not cap.isOpened():
            st.error("Error: Could not access webcam. Please check your camera permissions in System Settings.")
            st.info("To fix this on macOS:")
            st.info("1. Go to System Settings > Privacy & Security > Camera")
            st.info("2. Enable camera access for your browser/terminal")
            st.stop()

        # Add a stop button
        stop_button = st.button('Stop Webcam', key='stop_webcam')

        st.markdown("<div style='text-align: center; color: #4CAF50; font-weight: bold;'>Webcam is active! Detecting objects...</div>", unsafe_allow_html=True)
        
        while not stop_button:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from webcam")
                break
                
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform detection with confidence threshold
            results = model(frame, conf=confidence_threshold)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Display the frame
            stframe.image(annotated_frame, use_container_width=True)

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            fps_display.markdown(f"<div class='metric-container'><h3>FPS</h3><h2>{fps:.2f}</h2></div>", unsafe_allow_html=True)
            
            # Count and display detections
            num_detections = len(results[0].boxes)
            detection_count.markdown(f"<div class='metric-container'><h3>Detections</h3><h2>{num_detections}</h2></div>", unsafe_allow_html=True)

            # Check if the stop button was clicked
            if stop_button:
                break

        # Release the webcam
        cap.release()

    except Exception as e:
        st.error(f"Error accessing webcam: {str(e)}")
        st.info("To fix this on macOS:")
        st.info("1. Go to System Settings > Privacy & Security > Camera")
        st.info("2. Enable camera access for your browser/terminal")

else:  # Upload Video option
    # Add video upload option with more detailed instructions
    st.markdown("### Upload a Video File")
    st.markdown("Supported formats: MP4, MOV, AVI")
    
    video_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])

    if video_file is not None:
        # Show a progress bar while processing the video
        progress_text = "Preparing video for processing..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        # Get video info
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        st.markdown(f"<div style='text-align: center;'>Processing video with {total_frames} frames at {fps:.2f} FPS</div>", unsafe_allow_html=True)
        
        # Add a stop button
        stop_button = st.button('Stop Processing', key='stop_video')
        
        frame_count = 0
        while cap.isOpened() and not stop_button:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Update progress bar
            progress_bar.progress(min(frame_count / total_frames, 1.0), text=f"Processing frame {frame_count}/{total_frames}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame, conf=confidence_threshold)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, use_container_width=True)

            # Calculate and display FPS
            process_fps = 1.0 / (time.time() - start_time)
            fps_display.markdown(f"<div class='metric-container'><h3>Processing Speed</h3><h2>{process_fps:.2f} FPS</h2></div>", unsafe_allow_html=True)
            
            # Count and display detections
            num_detections = len(results[0].boxes)
            detection_count.markdown(f"<div class='metric-container'><h3>Detections</h3><h2>{num_detections}</h2></div>", unsafe_allow_html=True)
        
        cap.release()
        progress_bar.empty()
        
        if frame_count >= total_frames:
            st.success("Video processing complete!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Powered by YOLOv8 and Streamlit</p>
    <p>© 2023 Object Detection App</p>
</div>
""", unsafe_allow_html=True)