import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img as image_loading
from tensorflow.keras.preprocessing.image import img_to_array as extracting_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K


def prepare_data_modeling(path, df_type, VGGFace_model, size=(224,224)):

  x_type = list()
  y_type = list()
  character_foldes  =os.listdir(path+'/'+df_type+'/')
  integer_label_mapping = {}

  for index, character in enumerate(character_foldes):
    integer_label_mapping[index]=character
    image_path=os.listdir(path+'/'+df_type+'/'+character+'/')

    for sub_path in image_path:
      image_loaded     = image_loading(path+'/'+df_type+'/'+character+'/'+sub_path,target_size=size)
      arr_image        = extracting_array(image_loaded)
      expanded_img     = np.expand_dims(arr_image,axis=0)
      preprocess_img   = preprocess_input(expanded_img)
      vggWeights_img   = VGGFace_model(preprocess_img)
      transformed_img  = list(np.squeeze(K.eval(vggWeights_img)))
      x_type.append(transformed_img)
      y_type.append(index)
  if df_type == 'val':
    return np.array(x_type), np.array(y_type)
  else:
    return integer_label_mapping, np.array(x_type), np.array(y_type)
