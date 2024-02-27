#Importing Packages
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # pretrained YOLOv8x model

#input list of Images
src = ["C:\\Users\\vetri\\OneDrive\\Pictures\\Download\\cat.jpeg"]

def classify(image):

    # Run batched inference on a list of images
    results = model(image) 

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        # result.show()  # display to screen
        result.save(filename='result.jpg')  # save to disk
        # result.show()

classify('dog.jpeg')
    