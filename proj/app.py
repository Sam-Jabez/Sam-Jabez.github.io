import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('mask_detection_model.h5')

# Create a function to preprocess the input image
def preprocess(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to 224x224 pixels
    resized = cv2.resize(gray, (224, 224))
    # Expand the dimensions of the image to (1, 224, 224, 1)
    processed = np.expand_dims(resized, axis=0)
    processed = np.expand_dims(processed, axis=-1)
    return processed

# Create a VideoCapture object to capture video from the camera
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Detect faces in the frame using the OpenCV Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]
        # Preprocess the face ROI
        processed = preprocess(face_roi)
        # Use the pre-trained model to make predictions
        prediction = model.predict(processed)
        # Get the predicted class label (mask or no mask)
        label = "Mask" if prediction[0][0] > prediction[0][1] else "No Mask"
        # Draw a rectangle around the face
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Draw the predicted label and confidence score
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f'{max(prediction[0]):.2f}', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show the output frame
    cv2.imshow('Face Mask Detection', frame)
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
