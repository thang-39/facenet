#importing necessary libraries
import pickle
from tensorflow.keras.models import Model
import tensorflow as tf

#imorting necessary functions from ML_pipeline
from ML_pipeline.download_video import download_video
from ML_pipeline.create_new_folder import create_new_folder
from ML_pipeline.extracting_frames import extract_frames
from ML_pipeline.face_classifier import face_classifier_model
from ML_pipeline.vgg_prediction import vgg_video_predition
from ML_pipeline.vgg_prediction import vgg_image_prediction
from ML_pipeline.vgg_architecture import architecture
from ML_pipeline.data_prep_vgg import prepare_data_modeling

#run the following snippet only if video download and extraction of frames has to be done.
'''
# Download video from you tube
download_video(video_link = "https://www.youtube.com/watch?v=NzOTuh63eVs", path_to_store='../input/video')

# Create new folder to save frames 
create_new_folder(path = '../input/frames_path')

# Extract frames from video and saving them in frames_path folder
extract_frames(frames_path= '../input/frames_path/', video_file="../input/video/Friends - Monica and Chandlers Wedding.mp4")
'''

# Load the vgg model architecture
model = architecture()

# Load vggface weights
model.load_weights('../prebuilt_models/vgg_face_weights.h5')

# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
VGGFace=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
print('model is created')


# Preparing person mapping
path='../input'
character_mapping, train_x, train_y = prepare_data_modeling(path, 'train', VGGFace)
test_x, test_y = prepare_data_modeling(path, 'val', VGGFace)

# Saving person mapping
with open('../prebuilt_models/character_mapping.pkl', 'wb') as f:
    pickle.dump(character_mapping, f)

# Load person mapping pickle object
with open('../prebuilt_models/character_mapping.pkl', 'rb') as handle:
    person_map = pickle.load(handle)

# Model fitting
# only run below snipet if model fitting has to be done
save_path='../prebuilt_models'
face_classifier_model(train_x, train_y, test_x, test_y, save_path)

# Load saved model
face_recognition_classifier=tf.keras.models.load_model('../prebuilt_models/face_classifier_model.h5')

# VGG face prediction on frames
size_vgg= (224,224)
frames_path= '../input/test_frames/'
output='../output/'
vgg_image_prediction(frames_path= frames_path,output_path=output ,VGGFace_model= VGGFace, face_recognition_classifier= face_recognition_classifier, person_map= person_map, size=size_vgg )

# VGG face prediction on video
print('press Q on keyboard to exit from video window')
video='../input/video/Friends - Monica and Chandlers Wedding.mp4'
vgg_video_predition(video, frames_path, VGGFace, face_recognition_classifier, person_map)


