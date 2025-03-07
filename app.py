import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import platform
import time
import torch
import os
import socket
import logging

# Suppress OpenCV webcam errors in logs
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"  # Suppress OpenCV logs
logging.getLogger("opencv-python").setLevel(logging.FATAL)

# Initialize session state for source selection
if 'source_radio' not in st.session_state:
    st.session_state.source_radio = "Webcam"

# Detect user agent for browser-specific instructions
from streamlit.web.server.websocket_headers import _get_websocket_headers

try:
    user_agent = _get_websocket_headers().get("User-Agent", "")
    st.session_state['user_agent'] = user_agent
except:
    st.session_state['user_agent'] = ""

# Monkey patch torch.load to handle PyTorch 2.6+ compatibility
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # Force weights_only=False for compatibility
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load

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
        ["Webcam", "Upload Video"],
        key="source_radio"  # Use session state key
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

# Add a demo option for users who can't access webcam
demo_expander = st.expander("Can't access webcam? Try a demo video instead")
with demo_expander:
    st.markdown("""
    If you're having trouble accessing your webcam (especially on Streamlit Cloud), 
    you can try the "Upload Video" option in the sidebar and use a sample video.
    
    Here are some sample videos you can download and then upload:
    - [Sample Video 1: Street Traffic](https://github.com/ultralytics/assets/raw/main/examples/bus.mp4)
    - [Sample Video 2: People Walking](https://github.com/ultralytics/assets/raw/main/examples/person.mp4)
    
    Or you can use any video file from your device.
    """)
    if st.button("Switch to Video Upload Mode"):
        st.session_state.source_radio = "Upload Video"
        st.experimental_rerun()

# Load YOLOv8 model
@st.cache_resource
def load_model():
    # Set environment variable to disable PyTorch 2.6+ warnings
    os.environ['TORCH_WARN_WEIGHTS_ONLY'] = '0'
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
    # Add clear instructions about browser permissions
    st.info("""
    ### Important: Browser Camera Permissions
    This app requires camera access. Please ensure:
    1. You've allowed camera permissions in your browser
    2. No other applications are currently using your camera
    3. Your camera is properly connected and working
    
    If you're using Streamlit Cloud, make sure to click 'Allow' when prompted for camera access.
    """)
    
    # Add a webcam test button
    if st.button("Test Browser Camera Access"):
        # Use HTML/JavaScript to directly access the browser's camera
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
            <h3>Browser Camera Test</h3>
            <p>If you see your camera feed below, your browser camera is working correctly:</p>
            <video id="webcamTest" autoplay style="width: 100%; max-width: 500px; border: 1px solid #ddd;"></video>
        </div>
        
        <script>
            // JavaScript to access browser camera
            const video = document.getElementById('webcamTest');
            
            // Check if browser supports getUserMedia
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function(stream) {
                        video.srcObject = stream;
                        console.log("Camera access successful");
                    })
                    .catch(function(error) {
                        console.error("Camera access error:", error);
                        document.body.insertAdjacentHTML('beforeend', 
                            '<div style="color: red; margin-top: 10px;">Error accessing camera: ' + error.message + '</div>');
                    });
            } else {
                document.body.insertAdjacentHTML('beforeend', 
                    '<div style="color: red; margin-top: 10px;">Your browser does not support camera access</div>');
            }
        </script>
        """, unsafe_allow_html=True)
        
        st.info("""
        ### What to do if the camera test fails:
        
        1. **Check browser permissions**: Look for the camera icon in your browser's address bar
        2. **Try a different browser**: Chrome works best for webcam access
        3. **Refresh the page**: Sometimes a refresh can resolve permission issues
        4. **Check for other apps**: Make sure no other applications are using your camera
        """)
    
    try:
        # For macOS, try different camera indices
        if is_macos:
            camera_indices = [0, 1, 2]  # Try indices 0, 1, and 2 for macOS
            cap = None
            for idx in camera_indices:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    st.success(f"Successfully connected to camera at index {idx}")
                    break
                cap.release()
        else:
            # Try multiple camera indices for Windows/Linux as well
            camera_indices = [0, 1, 2, 3]  # Try more indices for Windows/Linux
            cap = None
            for idx in camera_indices:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    st.success(f"Successfully connected to camera at index {idx}")
                    break
                cap.release()

        if not cap or not cap.isOpened():
            st.error("Error: Could not access webcam through OpenCV. Trying browser-based access...")
            
            # Fallback to browser-based webcam access
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #0066cc;">Browser-Based Webcam Access</h3>
                <p>We'll try to access your webcam directly through your browser:</p>
                <video id="webcam" autoplay style="width: 100%; border-radius: 5px;"></video>
                <canvas id="canvas" style="display: none;"></canvas>
                <div id="status" style="margin-top: 10px; font-weight: bold;"></div>
                <button id="captureBtn" style="margin-top: 10px; padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">Capture Frame for Detection</button>
                <div id="detectionResults" style="margin-top: 15px;"></div>
            </div>
            
            <script>
                // JavaScript to access browser camera and perform detection
                const video = document.getElementById('webcam');
                const canvas = document.getElementById('canvas');
                const status = document.getElementById('status');
                const captureBtn = document.getElementById('captureBtn');
                const detectionResults = document.getElementById('detectionResults');
                let stream = null;
                
                // Check if browser supports getUserMedia
                if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                    status.innerHTML = "Requesting camera access...";
                    navigator.mediaDevices.getUserMedia({ video: true })
                        .then(function(mediaStream) {
                            stream = mediaStream;
                            video.srcObject = stream;
                            status.innerHTML = "✅ Camera connected! Click the button below to capture a frame for object detection.";
                            status.style.color = "green";
                        })
                        .catch(function(error) {
                            console.error("Camera access error:", error);
                            status.innerHTML = "❌ Error accessing camera: " + error.message;
                            status.style.color = "red";
                        });
                } else {
                    status.innerHTML = "❌ Your browser does not support camera access";
                    status.style.color = "red";
                }
                
                // Capture frame when button is clicked
                captureBtn.addEventListener('click', function() {
                    if (!stream) {
                        status.innerHTML = "❌ No camera stream available";
                        return;
                    }
                    
                    // Set canvas dimensions to match video
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    
                    // Draw video frame to canvas
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Convert canvas to data URL
                    const imageData = canvas.toDataURL('image/jpeg');
                    
                    // Send to Streamlit
                    detectionResults.innerHTML = "<p>Processing image for detection...</p>";
                    
                    // This would normally send the image to the server for processing
                    // For now, we'll just show a placeholder message
                    setTimeout(() => {
                        detectionResults.innerHTML = "<p>Detection would happen here with the server-side model</p>";
                    }, 1000);
                });
                
                // Clean up when component is unmounted
                window.addEventListener('beforeunload', () => {
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                    }
                });
            </script>
            """, unsafe_allow_html=True)
            
            st.info("""
            ### Browser-Based Webcam Notes:
            
            - This is a fallback method when OpenCV can't access your camera
            - You must allow camera permissions in your browser
            - Click the "Capture Frame" button to take a snapshot for detection
            - For full real-time detection, try running this app locally
            """)
            
            st.stop()

        # Add a stop button
        stop_button = st.button('Stop Webcam', key='stop_webcam')

        st.markdown("<div style='text-align: center; color: #4CAF50; font-weight: bold;'>Webcam is active! Detecting objects...</div>", unsafe_allow_html=True)
        
        # Add a frame counter and error counter for robustness
        frame_count = 0
        error_count = 0
        max_errors = 5  # Maximum consecutive errors before stopping
        
        while not stop_button:
            try:
                start_time = time.time()
                ret, frame = cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    error_count += 1
                    if error_count > max_errors:
                        st.error("Too many consecutive frame reading errors. Stopping webcam.")
                        break
                    continue
                
                # Reset error counter on successful frame
                error_count = 0
                frame_count += 1
                
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
                    
            except Exception as e:
                error_count += 1
                st.error(f"Error processing frame: {str(e)}")
                if error_count > max_errors:
                    st.error("Too many errors. Stopping webcam.")
                    break
                time.sleep(0.1)  # Short delay before retrying

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