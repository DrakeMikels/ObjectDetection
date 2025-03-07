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
    - [Sample Video 1: Street Traffic](https://github.com/ultralytics/assets/raw/main/examples/zidane.jpg)
    - [Sample Video 2: People Walking](https://github.com/ultralytics/assets/raw/main/examples/bus.jpg)
    
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
    
    # Check if running in Streamlit Cloud
    is_cloud = False
    try:
        # Try to detect if running in Streamlit Cloud by checking hostname
        hostname = socket.gethostname()
        is_cloud = "streamlit" in hostname.lower() or "heroku" in hostname.lower()
    except:
        pass
        
    if is_cloud:
        st.warning("""
        ### Running in Cloud Environment
        You're accessing this app from Streamlit Cloud. Webcam access in cloud environments:
        
        1. Requires HTTPS (which Streamlit Cloud provides)
        2. Needs explicit browser permission (look for camera icon in address bar)
        3. May be blocked by some corporate networks or VPNs
        
        If you continue to have issues, try the Upload Video option instead.
        """)
    
    # Add a webcam test option
    if st.button("Test Webcam Connection"):
        with st.spinner("Testing webcam connection..."):
            # Suppress OpenCV warnings during testing
            logging.getLogger("opencv-python").setLevel(logging.ERROR)
            
            # Try to connect to webcam
            test_indices = [0, 1, 2, 3]
            test_results = []
            found_working_camera = False
            
            for idx in test_indices:
                try:
                    test_cap = cv2.VideoCapture(idx)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret and frame is not None and frame.size > 0:
                            found_working_camera = True
                            test_results.append(f"✅ Camera index {idx}: Working")
                            # Show a single frame from this camera
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                    caption=f"Test frame from camera index {idx}", 
                                    width=300)
                        else:
                            test_results.append(f"⚠️ Camera index {idx}: Connected but can't read frames")
                    else:
                        test_results.append(f"❌ Camera index {idx}: Not available")
                except Exception as e:
                    test_results.append(f"❌ Camera index {idx}: Error - {str(e)}")
                finally:
                    if 'test_cap' in locals() and test_cap is not None:
                        test_cap.release()
            
            # Display all test results
            st.write("### Camera Test Results:")
            for result in test_results:
                st.write(result)
                
            if found_working_camera:
                st.success("Webcam test successful! At least one camera is working.")
            else:
                st.error("No working cameras found.")
                
                # Provide browser-specific guidance
                user_agent = st.session_state.get('user_agent', '')
                if 'Chrome' in user_agent:
                    st.info("""
                    ### Chrome-specific tips:
                    1. Click the camera icon in the address bar
                    2. Select "Allow" for camera access
                    3. Refresh the page
                    """)
                elif 'Firefox' in user_agent:
                    st.info("""
                    ### Firefox-specific tips:
                    1. Click the camera icon in the address bar
                    2. Select "Allow" for camera access
                    3. Refresh the page
                    """)
                elif 'Safari' in user_agent:
                    st.info("""
                    ### Safari-specific tips:
                    1. Check Safari > Settings > Websites > Camera
                    2. Allow camera access for this website
                    3. Refresh the page
                    """)
                
                # Offer fallback to video upload
                st.info("### Alternative: You can use the 'Upload Video' option instead")
                if st.button("Switch to Video Upload", key="switch_to_video"):
                    st.session_state.source_radio = "Upload Video"
                    st.experimental_rerun()
    
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
            st.error("Error: Could not access webcam. Please check your camera permissions.")
            
            # Provide platform-specific instructions
            if is_macos:
                st.info("### To fix this on macOS:")
                st.info("1. Go to System Settings > Privacy & Security > Camera")
                st.info("2. Enable camera access for your browser/terminal")
            elif platform.system() == 'Windows':
                st.info("### To fix this on Windows:")
                st.info("1. Go to Settings > Privacy & Security > Camera")
                st.info("2. Ensure 'Allow apps to access your camera' is turned on")
                st.info("3. Make sure your browser has camera permissions")
            else:  # Linux
                st.info("### To fix this on Linux:")
                st.info("1. Check browser settings for camera permissions")
                st.info("2. Ensure your user has access to the video device (usually in /dev/video*)")
                st.info("3. If using Streamlit Cloud, note that physical webcams aren't available - this is expected")
            
            # Offer fallback to video upload
            st.info("### Alternative: You can use the 'Upload Video' option instead")
            if st.button("Switch to Video Upload", key="switch_to_video_main"):
                st.session_state.source_radio = "Upload Video"
                st.experimental_rerun()
                
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