### YOLOv8 for Wildlife and MedSAM for Medical Imaging

In this blog post, we are going to explore and implement two cutting-edge models:

- **YOLOv8** algorithms with **instance segmentation and object tracking**, combined with OpenCV for tracking animals in the wild.
- **Segment Anything in Medical Images (MedSAM)** for comprehensive medical image analysis, including organs, lesions, and tissues.


## YOLOv8 for Wildlife Tracking

[![image-segmentation-and-object-tracking](URL-to-image)](https://vimeo.com/917944666 "YOLOv8 for Wildlife Tracking")

[![YOLOv8 for Wildlife Tracking](https://imgur.com/M3Lyzhz)](https://vimeo.com/917944666 "YOLOv8 for Wildlife Tracking")

Our approach integrates instance segmentation and object tracking to effectively monitor wildlife. This unified method aims to make a significant contribution to conservation efforts by providing precise and real-time tracking of animals in their natural habitats.

Let's begin by importing the necessary libraries and setting up the device configurations.

```py
!pip install ultralytics opencv-python-headless torch torchvision torchaudio

import cv2
import torch
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
```

Since I am training the model locally, we will utilize apple silicon GPU by importing torch.device('mps') and initializing the tracking history defaultdict. Following this, we initialize the YOLO model with the specified model name.

```py
device = torch.device('mps')
track_history = defaultdict(lambda: [])

model = YOLO("yolov8x-seg.pt")   # segmentation model
```

Now, we are going to utilize OpenCV for video processing, which includes reading, displaying, and writing video frames. First, we initialize the video capture and the video writer, and we obtain the video properties such as width `w`, height `h`, and frames per second `fps`.

```python3
# Initialize video capture and video writer
cap = cv2.VideoCapture("/Users/het/Desktop/wildlife-trim.mp4")
w, h, fps = (int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('instance-segmentation-object-tracking.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
```

Now, let's process each frame:

```python3
# Process each frame
while True:
    ret, im0 = cap.read()
    if not ret:
        print("Either the video frame is empty, or video processing has been successfully completed.")
        break
```

This part of the code enters a loop to process each frame of the video. If a frame cannot be read (`ret` is `False`), the loop breaks.

Now, after processing each frame, the YOLO model's `track` method is called to perform object detection, instance segmentation, and tracking. We measure the inference time to ensure the system's real-time performance by optimizing the input metrics.

```python3
# measure inference time
start_time = time.time()
annotator = Annotator(im0, line_width=2)
results = model.track(im0, persist=True)
inference_time = time.time() - start_time
```

We utilize the Annotator and colors utility to draw segmented outlines and labels on the video frames. If objects are detected, their masks and IDs are extracted.

```python3
# draw segmentation masks and bounding boxes with labels on the frame
if results[0].boxes.id is not None and results[0].masks is not None:
    masks = results[0].masks.xy
    track_ids = results[0].boxes.id.int().cpu().tolist()
    
    # extract class IDs
    class_ids = results[0].boxes.data[:, -1].int().cpu().tolist()
    
    for mask, track_id, class_id in zip(masks, track_ids, class_ids):
        class_name = model.names[class_id] # uses the class_ids to get the model names
        annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f"{track_id}: {class_name}")
        
        print(f"ID {track_id}: {w}x{h} {class_name}, {inference_time*1000:.1f}ms")
```
After processing, the annotated frame is written to the output video file and the frame is displayed. Finally, the code releases the video writer and video capture, and closes all OpenCV windows to free up resources.

```python3
# Writing and Displaying the Frame
out.write(im0)
cv2.imshow("instance-segmentation-object-tracking", im0)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Releasing Resources
out.release()
cap.release()
cv2.destroyAllWindows()
```

## Segment Anything in Medical Images (MedSAM)
