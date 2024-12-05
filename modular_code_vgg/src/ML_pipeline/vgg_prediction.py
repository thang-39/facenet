#importing necessary libararies
import numpy as np
import cv2
import os
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img as image_loading
from tensorflow.keras.preprocessing.image import img_to_array as extracting_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from ML_pipeline.image_modification import face_extract_using_CV

# Function to predict faces in farmes by using VGG Face model
def vgg_image_prediction(frames_path, output_path, VGGFace_model, face_recognition_classifier, person_map, size = (224,224)):
    print('Prediction on frames started')
    for image_path in os.listdir(frames_path):
        if image_path=='image_.jpg':
            continue
        image, faces = face_extract_using_CV(frames_path+image_path)
    
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            cv2.imwrite(frames_path+'image_.jpg',roi_color)
            image_loaded    = image_loading(frames_path+'image_.jpg', target_size= size)
            arr_image       = extracting_array(image_loaded)
            expanded_img    = np.expand_dims(arr_image,axis=0)
            preprocess_img  = preprocess_input(expanded_img)
            vggWeights_img  = VGGFace_model(preprocess_img)
            transformed_img = K.eval(vggWeights_img)
            per_prob        = face_recognition_classifier.predict(transformed_img)
            os.remove(frames_path+'image_.jpg')
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if np.max(per_prob) >=0.35:
                name=person_map[np.argmax(per_prob)]
                img=cv2.putText(image,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2,cv2.LINE_AA)
            else:
                img=cv2.putText(image,'others',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2,cv2.LINE_AA)
        
        cv2.imwrite(output_path+image_path+'_vgg_pred.jpg', img) 
    print('Predicted frames are stored in ',output_path,'folder')
    print('Prediction on frames ended!')
    


# Function to predict faces in video by using VGG Face model
def vgg_video_predition(video_file, frames_path, VGGFace, model, person_map ):
    print('Prediction on video started')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    size = (224,224)
    
# Create a VideoCapture object and read from input file
    video_object = cv2.VideoCapture(video_file)  

    if (video_object.isOpened()== False): 
        print("Error opening video  file")
    
    while(video_object.isOpened()):   

        content, frame = video_object.read()
        if content == True:
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
            faces = faceCascade.detectMultiScale(
                color,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64),
                flags=cv2.cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = frame[y:y + h, x:x + w]
                cv2.imwrite(frames_path+'image_.jpg',roi_color)
            
                image_loaded    = image_loading(frames_path+'image_.jpg', target_size = size)
                arr_image       = extracting_array(image_loaded)
                expanded_img    = np.expand_dims(arr_image,axis=0)
                preprocess_img  = preprocess_input(expanded_img)
                img_encode      = VGGFace(preprocess_img)
                embed           = K.eval(img_encode)
                per_prob        = model.predict(embed)
                os.remove(frames_path+'image_.jpg')
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if np.max(per_prob) >=0.35:
                    name=person_map[np.argmax(per_prob)]
                    img=cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2,cv2.LINE_AA)
                else:
                    img=cv2.putText(frame,'others',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2,cv2.LINE_AA)

            cv2.imshow('Video', img)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        else: 
            break
        
    # the video capture object
    video_object.release()
    cv2.destroyAllWindows()
    print('Prediction on video ended!')