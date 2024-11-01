import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from PIL import Image

# Define emotion classes (adjust according to your model's output classes)
emotion_classes = ['angry','disgusted','fearful','happy','neutral','sad','surprised']  # Example labels

# Load the MobileNetV2 model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, len(emotion_classes))  # Adjust for your number of classes
model.load_state_dict(torch.load('mobilenet_final.pth', map_location=device))
model = model.to(device)
model.eval()

# Define transformations to preprocess the frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),    # Resize to model's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

temprature = 2000

cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop and preprocess the face
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
        face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Preprocess and add batch dimension

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor) / temprature  # Apply temperature scaling
            probabilities = torch.nn.functional.softmax(output, dim=1) # Calculate scaled probabilities
            confidence, predicted = torch.max(probabilities, 1)
            emotion = emotion_classes[predicted.item()]
        

        # Draw a rectangle around the face and add the emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
