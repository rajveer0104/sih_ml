import numpy as np
import cv2 as cv
import os
from PIL import Image

def create_user(f_id, name):
    web = cv.VideoCapture(0)
    if not web.isOpened():
        print("Error: Could not open webcam")
        return
    web.set(3, 640)
    web.set(4, 480)

    faces = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    f_dir = 'dataset'
    f_name = name
    path = os.path.join(f_dir, f_name)

    if not os.path.isdir(path):
        os.makedirs(path)

    counter = 0
    while True:
        ret, img = web.read()
        if not ret or img is None:
            print("Failed to grab frame")
            break

        img = cv.flip(img, 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        multi_face = faces.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in multi_face:
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)
            counter += 1

            cv.imwrite("{}/{}.{}.{}{}".format(path, name, f_id, counter, ".jpg"),gray[y:y+h, x:x+w])

            cv.imshow("image", img)

        k = cv.waitKey(100) & 0xff
        if k == 27:     # ESC to stop
            break
        elif counter >= 50:   # Save 80 images max
            break

    web.release()
    cv.destroyAllWindows()

create_user(1,"Rajveer")
import cv2 as cv
import os
import numpy as np
from PIL import Image

# Function to train face recognizer
def train():

    # Folder where dataset images are stored
    database = 'dataset'

    # Get all sub-directories inside dataset (each sub-directory usually represents one person)
    img_dir = [x[0] for x in os.walk(database)][1::]

    # Create Local Binary Pattern Histogram (LBPH) Face Recognizer
    recognizer = cv.face.LBPHFaceRecognizer_create()

    # Load Haar cascade classifier for face detection
    detector = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Lists to store training data
    faceSamples = []  # stores cropped faces
    ids = []          # stores labels (IDs)

    # Loop through each person's folder inside dataset
    for path in img_dir:
        path = str(path)  # convert path to string

        # Get all image file paths in this folder (only jpg, jpeg, png)
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Loop through each image in the current folder
        for imagePath in imagePaths:

            # Open image and convert it to grayscale
            PIL_img = Image.open(imagePath).convert('L')

            # Convert image to numpy array (required by OpenCV)
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract ID from filename (example: User.1.1.jpg → id = 1)
            id = int(os.path.split(imagePath)[-1].split('.')[1])

            # Detect faces in the grayscale image
            faces = detector.detectMultiScale(img_numpy)

            # Loop through detected faces
            for (x, y, w, h) in faces:
                # Crop the face and add to training data
                faceSamples.append(img_numpy[y:y + h, x:x + w])

                # Add the corresponding ID for this face
                ids.append(id)

    # Train the recognizer with face samples and their corresponding IDs
    recognizer.train(faceSamples, np.array(ids))

    # Save the trained model to a file (so we don’t need to train again)
    recognizer.write('trainer.yml')

    # Print how many unique people (IDs) were trained
    print('\n[INFO] {0} faces trained. Exiting Program'.format(len(np.unique(ids))))

    # Return the count of trained persons
    return len(np.unique(ids))


# Run the training function
train()

import cv2 as cv

def recognize(names):
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    cascadePath = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    faceCascade = cv.CascadeClassifier(cascadePath)
    font = cv.FONT_HERSHEY_SIMPLEX

    face_count = 0
    cam = cv.VideoCapture(0)
    cam.set(3, 640)  # width
    cam.set(4, 480)  # height

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    name = ""  # last recognized name

    while True:
        ret, img = cam.read()
        img = cv.flip(img, 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH))
        )

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                id = names[id]
                conf_text = f"{round(100 - confidence)}%"
            else:
                id = "unknown"
                conf_text = f"{round(100 - confidence)}% - Not Recognized"

            if name == id:
                face_count += 1
                if face_count > 21:
                    face_count = -100
            else:
                name = id
                face_count = 0

            cv.putText(img, str(id), (x+5, y-5), font, 1, (255,0,0), 2)
            cv.putText(img, str(conf_text), (x+5, y+h-5), font, 1, (0,0,255), 2)

        cv.imshow("CAM1 ", img)

        k = cv.waitKey(30) & 0xff
        if k == 27:  # ESC to exit
            break

    cam.release()
    cv.destroyAllWindows()


recognize({1: "Rajveer",2:"Anurima",3:"Yash",4:"Arjya",5:"Mousumi"})


