# Real-Time Object Detection with YOLOv8

This Streamlit app performs real-time object detection using YOLOv8, allowing users to:
- Use their webcam for live object detection
- Upload videos for object detection
- Adjust confidence threshold for detections

## Local Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the app locally:
   ```
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Create a GitHub repository and push this code to it
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Deploy the app

## Troubleshooting Webcam Issues

If you're having trouble with webcam access:

### Browser Permissions
- Make sure your browser has permission to access your camera
- Look for the camera icon in your browser's address bar and ensure it's allowed
- Try using a different browser (Chrome works best with webcam access)

### System Permissions
- **Windows**: Go to Settings > Privacy & Security > Camera and ensure apps have access
- **macOS**: Go to System Settings > Privacy & Security > Camera and allow access for your browser
- **Linux**: Check your distribution's settings for camera permissions

### Connection Issues
- Ensure no other applications are using your webcam
- Try disconnecting and reconnecting your webcam
- Use the "Test Webcam Connection" button in the app to diagnose issues

### Streamlit Cloud Specific
- Streamlit Cloud requires HTTPS for webcam access
- The first time you access the webcam, you must allow the permission prompt
- Some corporate networks may block webcam access

## Notes

- The app automatically downloads the YOLOv8n model on first run
- Webcam functionality may require camera permissions in your browser
- For macOS users, you may need to grant camera permissions in System Settings 