# Dispatch Monitoring System

A computer vision-based AI system designed to monitor commercial kitchen dispatch areas by detecting and classifying trays and dishes, with model enhancement functionality driven by user feedback.

## System Overview
The Dispatch Monitoring System provides intelligent, real-time monitoring of commercial kitchen dispatch areas through a combination of computer vision and user interaction. It features:

 - Object Detection: Uses YOLOv11 to detect key items including trays and dishes in the dispatch area.

 - State Classification: Applies MobileNetV2 to classify each detected item as empty, kakigori, or not_empty.

 - Integrates DeepSORT (Deep Simple Online and Realtime Tracking) to assign and maintain consistent object IDs across video frames, enabling precise tracking over time.

 - User Feedback Loop: Enables users to provide feedback on incorrect predictions to support future model retraining and performance improvement.

 - Web-Based Interface: Provides an intuitive, browser-accessible platform to upload videos, view tracking results, download processed output, and submit feedback.

## Core technologies

| Component        | Technology       |
|------------------|------------------|
| Detection         | YOLOv11         |
| Classification    | MobileNetV2     |
| Tracking          | DeepSORT        |
| Web Interface     | Flask           |
| Deployment        | Docker Compose  |

## Detection 
 - Base Model: ``` yolo11n.pt```
 - Dataset: custom labeled dish/tray dataset
 - Training detailes:
   - Image size: 640x640
   - Batch size: 16
   - Epochs: 100
 - Output: Exported ```best.pt``` used in deployment (```models/detection/best.pt```)
## Classification model ( MobileNetV2)

 - Based model: pre-trained ```MobileNetV2``` ( ImageNet weights)
 - Target class: ```empty```, ```kakigori```, ```not_empty```
 - Image size: 224x224
 - Preprocessing:
   - Normalized pixel values to [0,1]
   - Resized and shuffled
 - Training configuration:
   - Batch size: 32
   - Epoch 10
   - Loss function: Sparse categorical cross-entropy
 - Results: accuracy 0.94

## Installation and deployment
You can run the Dispatch Monitoring System either locally using Python or via Docker Compose for easy deployment.

### Python environment
1. Clone repository

   ```git clone https://github.com/diemmylelelele/dispatch_monitoring_system.git```

   ```cd dispatch_monitoring_system```

2. Set Up a Python Virtual Environment

   ``` python -m venv venv```

   ``` source venv/bin/activate ```

3. Install dependencies

   ``` pip install -r requirements.txt```

4. Run the app

   ``` python app.py```

   The Flask app will start at:
   http://localhost:5000

### Docker Compose Deployment

1. Clone repository

   ```git clone https://github.com/diemmylelelele/dispatch_monitoring_system.git```

   ```cd dispatch_monitoring_system```

2. Build and start the containers
   
   ``` docker-compose up --build ```

3. Acess the app

   Go to:
   http://localhost:5000

3. Stop the app

   To stop and remove containers: ``` docker-compose down```

## Usage instructions
### Web interface operations
1. Upload an .mp4 video for tracking

2. Wait for processing to complete

3. View the tracked video and

4. Submit corrections if any labels are incorrect

Here is the demo usage:





