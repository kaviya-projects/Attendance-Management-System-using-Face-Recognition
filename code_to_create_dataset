import cv2
import os

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

face_id = input('Enter your ID: ')

# Start capturing video
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize sample face image count
count = 0

# Ensure the dataset directory exists
assure_path_exists("dataset")

# Start looping
while True:
    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loop through each face detected
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Increment sample face image count
        count += 1

        # Save the captured image into the dataset folder
        cv2.imwrite(f"dataset/User.{str(face_id)}.{str(count)}.jpg", gray[y:y + h, x:x + w])

        # Display the video frame with rectangle
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If 50 images are taken, stop capturing
    elif count >= 50:
        print("Successfully Captured")
        break

# Stop video capture
vid_cam.release()

# Close all windows
cv2.destroyAllWindows()


