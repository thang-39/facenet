import cv2
from numpy import asarray
from skimage.transform import resize
from os import listdir
from os.path import isdir

#This function is used to extracted face from an image and the output is the raw image along with the face/faces.

def face_extract_using_CV(path):

    image = cv2.imread(path)
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
    color,
    scaleFactor=1.2,
    minNeighbors=10,
    minSize=(64,64),
    flags=cv2.CASCADE_SCALE_IMAGE
    )

    return image, faces


#This function is used to take face image and outputs array of pixels

def resize_face(filename, size):
    
    image = cv2.imread(filename)
    resize_image = resize(image, size)
    pixels = asarray(resize_image)

    return pixels

#This function is used to take person's faces from the respective folders and pass on to face modeling function

def faces_from_folders(folder, size):
    
    extracted_faces = []
    
    for filename in listdir(folder):
        image_path = folder + filename
        face = resize_face(image_path, size)
        extracted_faces.append(face)
        
    return extracted_faces


#This function is used to iterate over each person's face from the respective folders and output an array of person's face pixels and the corresponding label

def loading_images(folder, size):
    X = [] ; y = []
    
    for sub_folder in listdir(folder):
        
        image_path = folder + sub_folder + '/'
        if not isdir(image_path):
            continue
        faces = faces_from_folders(image_path, size)
        labels = [sub_folder for _ in range(len(faces))]
        
        print('Loaded %d samples for character: %s' % (len(faces), sub_folder))
        
        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)