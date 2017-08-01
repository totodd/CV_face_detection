import face_recognition             # Face Recognition Library
import cv2                          # OpenCV Library
from skimage import io              # Library to Save Image On Disk
import pickle                       # Library to Save Training Face Features
from sklearn import svm             # Library qfor SVM Classifier
import numpy as np                  # Library for Matrix Manipulation
import time                         # Library to calculate Time performance
import os                           # Library to read folder path
import matplotlib.pyplot as plt     # Library to plot images
from PIL import Image, ImageDraw    # Library to Draw Landmarks


# Function to convert jpg image into array
def jpg_image_to_array(image):
    """ Loads JPEG image into 3D Numpy array of shape (width, height, channels) """
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr


# Load Generated Face Encodings
f = open('store.pckl', 'rb')
store_object = pickle.load(f)
f.close()


# Database Paths
path = './output/'
path_trainingData = './training-set/'
path_ref = './reference/'


# Time Initialisation
startT = time.time()


# Load Ground Truth into Memory
sequence = []
truth = []
with open(path_ref + 'GroundTruth_class01.csv', 'r') as myFile:
    for line in myFile:
        lis = line.split(',')
        sequence.append(lis[0])
        truth.append(lis[1])


print '>>>>> STEP 1 : Loading Pre-Extracted Features of Training Set {0:.2f}s'.format((time.time() - startT) % 60)
face_encodings = store_object[0]
face_encodings_label = store_object[1]
print '               training face-feature matrix: [ {} x {} ]'.format(len(face_encodings), len(face_encodings[0]))
print '               training face-label matrix: [ {} x {} ]'.format(len(face_encodings_label), len(face_encodings_label[0][0]))


print '>>>>> STEP 2 : Training SVM --------------------------------- {0:.2f}s'.format((time.time() - startT) % 60)
train_X = np.array(face_encodings)
train_Y = np.array(face_encodings_label)
clf = svm.NuSVC()
clf.fit(train_X, train_Y)
print '               Using Non-linear SVM'


print '>>>>> STEP 3 : Loading Unknown Image ------------------------ {0:.2f}s'.format((time.time() - startT) % 60)
unknown_image = face_recognition.load_image_file(path_ref + "anuclass01.JPG")
# Marvin_image = face_recognition.load_image_file(path + "Marvin.jpg")
# Tao_image = face_recognition.load_image_file(path + "Tao.jpg")

# print Tao_image.shape
# print unknown_image.shape

# # Get the face encodings for each face in each image file
# # Since there could be more than one face in each image, it returns a list of encordings.
# # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
# Marvin_face_encoding = face_recognition.face_encodings(Marvin_image)[0]
# Tao_face_encoding = face_recognition.face_encodings(Tao_image)[0]
# unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
# num = len(face_recognition.face_encodings(unknown_image))
# print unknown_face_encoding.shape


print '>>>>> STEP 4 : Finding Faces -------------------------------- {0:.2f}s'.format((time.time() - startT) % 60)
# face_locations = face_recognition.face_locations(unknown_image)
f = open('storeFace_Location.pckl', 'rb')
face_locations = pickle.load(f)
f.close()
print '               Find frontal faces using "dlib" library'
print '               {} faces are found in the unknown image'.format(len(face_locations))


print '>>>>> STEP 5 : Extract Features from Face -------------------- {0:.2f}s'.format((time.time() - startT) % 60)
top = face_locations[0][0]
right = face_locations[0][1]
bottom = face_locations[0][2]
left = face_locations[0][3]
image = unknown_image[top:bottom, left:right]
face_landmarks_list = face_recognition.face_landmarks(image)
num = 0
print '               Find landmarks of faces using "dlib" model : shape_predictor_68_face_landmarks.dat'
print '               landmarks on each part'
for cat in face_landmarks_list[0]:
    print '               {} : {}'.format(cat, len(cat))
    num = num + len(cat)
print '               Each face has total {} landmarks'.format(num)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
print '               Find crucial features of faces using "dlib" model from landmarks: dlib_face_recognition_resnet_model_v1.dat'
print '               Feature matrix: [ {} x {} ]'.format(len(face_encodings), len(face_encodings[0]))
# print face_encodings[0]


######################## comparing faces ########################
print '>>>>> STEP 6 : Predict Faces Using SVM ---------------------- {0:.2f}s'.format((time.time() - startT) % 60)
total_num = 0
correct_num = 0
face_names = []
flag = []
for i, face_encoding in enumerate(face_encodings):
    # See if the face is a match for the known face(s)
    predict_label = clf.predict(face_encoding.reshape(1, -1))

    name = predict_label[0]

    # print truth[i]
    if truth[i] != 'NG':
        total_num = total_num + 1
        if name == truth[i]:
            correct = 1
            correct_num = correct_num + 1
        else:
            correct = 0
    else:
        name = 'NG'
        correct = 2

    face_names.append(name)
    flag.append(correct)


######################## overlay on the big image ################################
print '>>>>> STEP 7 : Drawing Bounding Box on the Unknown Image ------ {0:.2f}s'.format((time.time() - startT) % 60)
for (top, right, bottom, left), name, flg in zip(face_locations, face_names, flag):
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    # top *= 1
    # right *= 1
    # bottom *= 1
    # left *= 1

    # Draw a box around the face
    if flg == 1:
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(unknown_image, (left, bottom - 10), (right, bottom), (0, 255, 0), cv2.FILLED)
    elif flg == 0:
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.rectangle(unknown_image, (left, bottom - 10), (right, bottom), (255, 0, 0), cv2.FILLED)
    elif flg == 2:
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(unknown_image, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)

    # Draw a label with a name below the face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(unknown_image, name, (left + 12, bottom - 12), font, 0.6, (255, 255, 255), 1)

######################## plot cropped faces #######################
fig = plt.figure()
row_num = 7
qty = len(face_locations)
for i, (top, right, bottom, left) in enumerate(face_locations):
    fig.add_subplot(row_num, (qty * 2 / row_num + 1), 2 * i + 1)
    buff = 20
    image = unknown_image[top - buff:bottom + buff, left - buff:right + buff]
    plt.imshow(image)

    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image, 'RGBA')
        # print face_landmarks['right_eyebrow']
        #     # Make the eyebrows into a nightmare
        for points in face_landmarks:
            # print points
            d.line(face_landmarks[points], fill=(255, 255, 255, 255), width=5)

    fig.add_subplot(row_num, (qty * 2 / row_num + 1), 2 * i + 2)
    pic = jpg_image_to_array(pil_image)
    plt.imshow(pic)

##################### draw a face beside the face_location #####################
for i, (top, right, bottom, left) in enumerate(face_locations):
    width = right - left
    height = bottom - top
    if face_names[i] != 'NG':
        addr = path_trainingData + face_names[i]
        f = os.listdir(addr)[0]
        truth_image = face_recognition.load_image_file(addr + '/' + f)
    else:
        truth_image = face_recognition.load_image_file(path_ref + "todd.jpg")
    # truth_image = face_recognition.load_image_file(path + "Tao.jpg")
    truth_image = cv2.resize(truth_image, (width, height))
    final_image = unknown_image
    final_image[top:(top + height), (left + width):(left + 2 * width)] = truth_image

print '>>>>> STEP 8 : saving image as <classified.jpg> --------------- {0:.2f}s'.format((time.time() - startT) % 60)
io.imsave(path + "classified.jpg", unknown_image)
print '               total number of present people: 55'
print '               total number of face detected: {}'.format(len(face_locations))
print '               detection rate: {}%'.format(100 * len(face_locations) / 55)
print '               total number of face with training data: {}'.format(total_num)
print '               total number of correct face recognized: {}'.format(correct_num)
print '               correct rate: {}%'.format(100 * correct_num / total_num)

plt.show()
