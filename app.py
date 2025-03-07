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
import base64
from PIL import Image
import io

# Suppress OpenCV webcam errors in logs
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"  # Suppress OpenCV logs
logging.getLogger("opencv-python").setLevel(logging.FATAL)

# Initialize session state for source selection
if 'source_radio' not in st.session_state:
    st.session_state.source_radio = "Webcam"

# Initialize session state for webcam image
if 'webcam_image' not in st.session_state:
    st.session_state.webcam_image = None

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
    1. You've allowed camera permissions in your browser when prompted
    2. No other applications are currently using your camera
    3. Your camera is properly connected and working
    
    If you're using Streamlit Cloud, you should see a camera permission prompt from your browser.
    """)
    
    # Use our custom webcam component
    captured_image_placeholder, detection_results_placeholder = webcam_component()
    
    # Add a callback for processing captured frames
    if 'webcam_image' in st.session_state and st.session_state.webcam_image:
        # Get the image data from session state
        img_data = st.session_state.webcam_image
        
        try:
            # Convert base64 image to numpy array
            img_data = img_data.split(",")[1]  # Remove the data URL prefix
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
            
            # Display the captured image
            captured_image_placeholder.image(img_array, caption="Captured Frame", use_column_width=True)
            
            # Perform detection
            results = model(img_array, conf=confidence_threshold)
            
            # Visualize the results
            annotated_frame = results[0].plot()
            
            # Display the results
            detection_results_placeholder.image(annotated_frame, caption="Detection Results", use_column_width=True)
            
            # Display detection count
            num_detections = len(results[0].boxes)
            st.success(f"Detected {num_detections} objects in the captured frame")
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Add JavaScript to handle the custom event
    st.markdown("""
    <script>
    // Listen for the custom capture event
    window.addEventListener('streamlit:captureFrame', function(event) {
        // Get the image data
        const imageData = event.detail.data;
        
        // Send to Streamlit via the Streamlit component
        if (window.Streamlit) {
            window.Streamlit.setComponentValue(imageData);
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
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

# Add a custom webcam component
def webcam_component():
    """Create a custom webcam component that works in Streamlit Cloud"""
    # Generate a unique ID for this component instance
    component_id = "webcam_" + str(int(time.time()))
    
    # Create a placeholder for the webcam feed
    webcam_placeholder = st.empty()
    
    # Create placeholders for the captured image and detection results
    captured_image_placeholder = st.empty()
    detection_results_placeholder = st.empty()
    
    # HTML/JS for the webcam component
    webcam_html = f"""
    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <h3 style="color: #4CAF50;">Browser Webcam Access</h3>
        <video id="{component_id}_video" autoplay playsinline style="width: 100%; max-width: 640px; border-radius: 5px;"></video>
        <div id="{component_id}_status" style="margin: 10px 0; font-weight: bold;"></div>
        <button id="{component_id}_capture" style="background-color: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-right: 10px;">Capture for Detection</button>
        <button id="{component_id}_stop" style="background-color: #f44336; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; display: none;">Stop Camera</button>
        
        <canvas id="{component_id}_canvas" style="display: none;"></canvas>
        <input type="hidden" id="{component_id}_data" name="data" value="">
    </div>
    
    <script>
        // Wait for DOM to load
        document.addEventListener('DOMContentLoaded', (event) => {{
            const videoElement = document.getElementById('{component_id}_video');
            const statusElement = document.getElementById('{component_id}_status');
            const captureButton = document.getElementById('{component_id}_capture');
            const stopButton = document.getElementById('{component_id}_stop');
            const canvasElement = document.getElementById('{component_id}_canvas');
            const dataElement = document.getElementById('{component_id}_data');
            
            let stream = null;
            
            // Function to start the webcam
            async function startWebcam() {{
                statusElement.textContent = "Requesting camera access...";
                statusElement.style.color = "blue";
                
                try {{
                    // Request camera access with preferred settings
                    stream = await navigator.mediaDevices.getUserMedia({{
                        video: {{ 
                            width: {{ ideal: 640 }},
                            height: {{ ideal: 480 }},
                            facingMode: "user"
                        }},
                        audio: false
                    }});
                    
                    // Set the video source to the camera stream
                    videoElement.srcObject = stream;
                    
                    // Show success message
                    statusElement.textContent = "✅ Camera connected! Click 'Capture for Detection' to analyze a frame.";
                    statusElement.style.color = "green";
                    
                    // Show stop button
                    stopButton.style.display = "inline-block";
                    
                }} catch (error) {{
                    // Show error message
                    statusElement.textContent = "❌ Error accessing camera: " + error.message;
                    statusElement.style.color = "red";
                    console.error("Camera access error:", error);
                }}
            }}
            
            // Function to stop the webcam
            function stopWebcam() {{
                if (stream) {{
                    stream.getTracks().forEach(track => track.stop());
                    videoElement.srcObject = null;
                    stream = null;
                    statusElement.textContent = "Camera stopped. Refresh the page to restart.";
                    statusElement.style.color = "orange";
                    stopButton.style.display = "none";
                }}
            }}
            
            // Function to capture a frame
            function captureFrame() {{
                if (!stream) {{
                    statusElement.textContent = "❌ No camera stream available. Please start the camera first.";
                    statusElement.style.color = "red";
                    return;
                }}
                
                // Set canvas dimensions to match video
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                
                // Draw video frame to canvas
                const ctx = canvasElement.getContext('2d');
                ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                
                // Convert canvas to data URL (JPEG format)
                const imageData = canvasElement.toDataURL('image/jpeg', 0.9);
                
                // Store the image data in the hidden input
                dataElement.value = imageData;
                
                // Send to Streamlit
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: imageData
                }}, '*');
                
                // Update status
                statusElement.textContent = "Frame captured! Processing...";
                statusElement.style.color = "blue";
            }}
            
            // Add event listeners
            captureButton.addEventListener('click', captureFrame);
            stopButton.addEventListener('click', stopWebcam);
            
            // Start webcam automatically
            startWebcam();
        }});
    </script>
    """
    
    # Display the webcam component
    webcam_placeholder.markdown(webcam_html, unsafe_allow_html=True)
    
    # Create a component to receive the captured image
    from streamlit.components.v1 import components
    
    def capture_callback(image_data):
        if image_data:
            st.session_state.webcam_image = image_data
            return True
        return False
    
    # Register the component
    custom_component = components.declare_component(
        "webcam_capture",
        render_func=lambda: None  # We're using the HTML/JS above
    )
    
    # Use the component to receive the captured image
    result = custom_component(key=f"webcam_capture_{component_id}", default=None)
    if result:
        st.session_state.webcam_image = result
        st.experimental_rerun()
    
    # Return the placeholders for later use
    return captured_image_placeholder, detection_results_placeholder