import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def face_classifier_model(train_x, train_y, test_x, test_y, path):
    #model architecture
    face_recognition_classifier=Sequential()
    face_recognition_classifier.add(Dense(units=120,input_dim=train_x.shape[1],activation = 'tanh'))
    face_recognition_classifier.add(BatchNormalization())
    face_recognition_classifier.add(Dropout(0.25))
    face_recognition_classifier.add(Dense(units=50,activation = 'tanh'))
    face_recognition_classifier.add(BatchNormalization())
    face_recognition_classifier.add(Dropout(0.25))
    face_recognition_classifier.add(Dense(units=10,activation = 'tanh'))
    face_recognition_classifier.add(BatchNormalization())
    face_recognition_classifier.add(Dropout(0.25))
    face_recognition_classifier.add(Dense(units=5,activation = 'softmax'))

    face_recognition_classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])
    
    #model fitting
    ckpt = ModelCheckpoint(path, monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_accuracy", mode="max", patience=2)

    face_recognition_classifier.fit(train_x,train_y,epochs=10,validation_data=(test_x,test_y),callbacks=[ckpt, early])
    
    save_model(face_recognition_classifier, path+'/face_classifier_model.h5')
    print('Model Saved in ', path+'/face_classifier_model.h5')
    #return face_recognition_classifier

