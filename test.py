import streamlit as st
import cv2
import torch
import numpy as np

# Path to the YOLOv5 model weights
model_path = 'best.pt'

# Load the YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

model = load_model()

# Function to process video frames and perform detection
def process_frame(frame):
    results = model(frame)
    frame = np.squeeze(results.render())
    return frame

# Streamlit app
def main():
    st.title("Real-Time Object Detection with YOLOv5")

    # Initialize variables for start and stop detection
    start_detection = st.button("Start Detection")
    stop_detection = st.button("Stop Detection")

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    detecting = False

    while True:
        if start_detection:
            detecting = True

        if stop_detection:
            detecting = False

        if detecting:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video")
                break

            # Resize frame
            frame = cv2.resize(frame, (1020, 700))

            # Process the frame for detection
            frame = process_frame(frame)

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            stframe.image(frame, channels="RGB", use_column_width=True)

        # Break the loop if the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
