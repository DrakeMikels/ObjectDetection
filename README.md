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

## Notes

- The app automatically downloads the YOLOv8n model on first run
- Webcam functionality may require camera permissions in your browser
- For macOS users, you may need to grant camera permissions in System Settings 