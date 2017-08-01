import face_recognition
import cv2
# import matplotlib.pyplot as plt
# from skimage import io
# import os
from sklearn import svm
import numpy as np
import pickle
import os
from PIL import Image, ImageDraw


f = open('store.pckl', 'rb')
store_object = pickle.load(f)
f.close()

# Load the jpg files into numpy arrays
path = './output/'
path_trainingData = './training-set/'
path_ref = './reference/'


face_encodings = store_object[0]
face_encodings_label = store_object[1]

train_X = np.array(face_encodings)

train_Y = np.array(face_encodings_label)


clf = svm.NuSVC()
clf.fit(train_X, train_Y)



# Function to convert jpg image into array
def jpg_image_to_array(image):
    """ Loads JPEG image into 3D Numpy array of shape (width, height, channels) """
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr
# Marvin_image = face_recognition.load_image_file(path + "Marvin.jpg")
# Marvin_face_encoding = face_recognition.face_encodings(Marvin_image)[0]
#
# Tao_image = face_recognition.load_image_file(path + "Tao.jpg")
# Tao_face_encoding = face_recognition.face_encodings(Tao_image)[0]

# predict_label = clf.predict(Marvin_face_encoding.reshape(1, -1))
# predict_label2 = clf.predict(Tao_face_encoding.reshape(1, -1))
#
# print predict_label
# print predict_label2

#######################################################################
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# video_capture = cv2.VideoCapture(path + 'clear.avi')

# Load a sample picture and learn how qto recognize it.
Marvin_image = face_recognition.load_image_file(path_ref + "Marvin.jpg")
Tao_image = face_recognition.load_image_file(path_ref + "Tao.jpg")
Kim_image = face_recognition.load_image_file(path_ref + "kim.jpg")
Weikun_image = face_recognition.load_image_file(path_ref + "Weikun.jpg")
Raja_image = face_recognition.load_image_file(path_ref + "Raja.jpg")

Marvin_face_encoding = face_recognition.face_encodings(Marvin_image)[0]
Tao_face_encoding = face_recognition.face_encodings(Tao_image)[0]
Kim_face_encoding = face_recognition.face_encodings(Kim_image)[0]
Weikun_face_encoding = face_recognition.face_encodings(Weikun_image)[0]
Raja_face_encoding = face_recognition.face_encodings(Raja_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # print type(frame)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # print frame.shape
    # print small_frame.shape
    #small_frame = frame

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces([Kim_face_encoding, Weikun_face_encoding, Raja_face_encoding], face_encoding, tolerance=0.5)
            # match1 = face_recognition.compare_faces([Tao_face_encoding], face_encoding)
            if match[0]:
                name = 'Kim'
            elif match[1]:
                name = 'Weikun'
            elif match[2]:
                name = 'Raja V2'


            else:
                predict_label = clf.predict(face_encoding.reshape(1, -1))
            # match = face_recognition.compare_faces([Tao_face_encoding], face_encoding)
            # name = "Unknown"
            #
            # if match[0]:
                name = str(predict_label[0])

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # ##################### draw a face beside the face_location #####################
    for i, (top, right, bottom, left) in enumerate(face_locations):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        width = right - left
        height = bottom - top
        if face_names[i] != 'NG':
            if face_names[i] == 'Kim':
                truth_image = face_recognition.load_image_file(path_ref + 'kim.jpg')
            elif face_names[i] == 'Weikun':
                truth_image = face_recognition.load_image_file(path_ref + 'Weikun.jpg')
            elif face_names[i] == 'Raja V2':
                truth_image = face_recognition.load_image_file(path_ref + 'Raja.jpg')
            else:
                addr = path_trainingData + face_names[i]
                f = os.listdir(addr)[0]
                truth_image = face_recognition.load_image_file(addr + '/' + f)

        else:
            truth_image = face_recognition.load_image_file(path_ref + "todd.jpg")
        # truth_image = face_recognition.load_image_file(path + "Tao.jpg")
        truth_image = cv2.resize(truth_image, (width, height))
        # frame[top:(top + height), (left + width):(left + 2 * width)] = truth_image[0:height, 0:width]
        window = frame[top:(top + height), (left + width):(left + 2 * width)]
        frame[top:(top + height), (left + width):(left + 2 * width)] = truth_image[0: window.shape[0], 0: window.shape[1]]

    # face_landmarks_list = face_recognition.face_landmarks(frame)

    # for face_landmarks in face_landmarks_list:
    #     pil_image = Image.fromarray(frame)
    #     d = ImageDraw.Draw(pil_image, 'RGBA')
    #
    #     d.polygon(face_landmarks['top_lip'], fill=(0, 0, 150, 128))
    #     d.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 150, 128))
    #     d.line(face_landmarks['top_lip'], fill=(0, 0, 150, 64), width=8)
    #     d.line(face_landmarks['bottom_lip'], fill=(0, 0, 150, 64), width=8)
    #     d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    #     d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
    #     d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(150, 150, 150, 110), width=6)
    #     d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(150, 150, 150, 110), width=6)

    # pil_image = Image.fromarray(frame)
    # for face_landmarks in face_landmarks_list:
    #
    #     d = ImageDraw.Draw(pil_image, 'RGBA')
    #     # print face_landmarks['right_eyebrow']
    #     #     # Make the eyebrows into a nightmare
    #     for points in face_landmarks:
    #         # print points
    #         d.line(face_landmarks[points], fill=(255, 255, 255, 255), width=5)

    # pic = jpg_image_to_array(pil_image)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()