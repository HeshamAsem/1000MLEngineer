# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:18:56 2022

@author: Ahmed KASAPBAŞI
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.metrics  import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from  tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import confusion_matrix

import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
import glob as gb
import pandas as pd
import cv2


import splitfolders # or import splitfolders



###################################spliting Data###################################


input_folder = "G:/03 Leaves_orginal10/Grapevine_Leaves_Image_Dataset/"
output = "G:/03 Leaves_orginal10/divided10" #where you want the split datasets saved. one will be created if it does not exist or none is set
if not(os.path.exists(output)):
    #os.rmdir(photorgb)
    os.mkdir(output)

if len(os.listdir(output)) != 0:
        # removing the file using the os.remove() method
        shutil.rmtree(output)

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.7, 0.2, .1)) # ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.

        
        
        
        
###############################################################################


       

##############################################################################
train_path=output+'/train/'
valid_path=output+'/val/'
test_path=output+'/test/'

print('---------------------reading training data-----------------------')
for folder in os.listdir(train_path):
    files=gb.glob(pathname=str(train_path+folder+'/*.png')) #البحث عن ملفات بلاحقة محددة ضمن مجلد
    print(f'for training data,found {len(files)} in folder {folder}')
    
print('---------------------reading valid data-----------------------')


for folder in os.listdir(valid_path):
    files=gb.glob(pathname=str(valid_path+folder+'/*.png'))
    print(f"for validing data, found {len(files)} in folder {folder}")
    
print('---------------------reading testing data-----------------------')

for folder in os.listdir(test_path):
    files=gb.glob(pathname=str(test_path+folder+'/*.png'))
    print(f"for testing data, found {len(files)} in folder {folder}")


print('---------------------طباععة أبعاد الصور  وأعدادها لكل بعد موجود-----------------------\n\n\n')
def sizeOfData(path):

    size=[]
    
    folder=os.listdir(path)
    
    
    for i in folder:
        files=gb.glob(pathname=str(path+i+'/*.png'))
        for file in files:
            image=plt.imread(file)
            size.append(image.shape)

    return [pd.Series(size).value_counts()]

print('-------------------printing training Dataset Dimensions and numbers ----------------------')

print(sizeOfData(train_path))

print('-------------------printing vald Dataset Dimensions and numbers ----------------------')
print(sizeOfData(valid_path))


print('-------------------printing testing Dataset Dimensions and numbers ----------------------')
print(sizeOfData(test_path))


print("================Reading Images=================")


code = {'Ak':0 ,'Ala_Idris':1,'Buzgulu':2,'Dimnit':3,'Nazli':4}


def getcode(n):
    for x,y in code.items():
        #print(x,y)
        if n==y:
            return x
        elif n==x:
            print(y)
            return y
s=224
def ReadingImages(path,s):
    xdata=[]
    ydata=[]
    
    for folder in os.listdir(path):
       
        files=gb.glob(pathname=str(path+folder+'/*.png'))
        for file in files:
            img=cv2.imread(file)
            img=cv2.resize(img,(s,s))
            xdata.append(list(img))
            ydata.append(code[folder])
                
    print(f"we have {len(xdata)} items in xdata {path[-6:-1]}")
    plt.figure(figsize=(20,20))
    for n,i in enumerate(list(np.random.randint(0,len(xdata),36))):
        plt.subplot(6,6,n+1)
        plt.imshow(xdata[i])
        plt.axis('off')
        
        plt.title(getcode(ydata[i]))
    
    return xdata,ydata
   
    
        
    
xtrain,ytrain=ReadingImages(train_path,s)
xtest,ytest=ReadingImages(valid_path, s)
#xpred=ReadingImages(test_path, s)


#################################The MODEL Training#################################

train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path,target_size=(224,224),batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path,target_size=(224,224),batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path,target_size=(224,224),batch_size=10,shuffle=False)


mobile=tf.keras.applications.mobilenet.MobileNet()
mobile.summary()

x=mobile.layers[-6].output

output=Dense(units=5,activation='softmax')(x)
model=Model(inputs=mobile.input,outputs=output)

for layer in model.layers[:-23]:
    layer.trainable=False
    
model.summary()

model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])


hist=model.fit(x=train_batches,steps_per_epoch=len(train_batches),# there is no batch size here
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=60,
          verbose=1
          )

###############################Saving the model###############################

import h5py
model.save('G:/03 Leaves_orginal10/pictures_training_Adam_60_epoch_0.00001/Trained_model60epochs.h5')


############################summarize history for accuracy############################

plt.plot(hist.history['accuracy'], label = 'train')
plt.plot(hist.history['val_accuracy'], label = 'val')
plt.title('Leaves model : Training &  Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()


plt.plot(hist.history['loss'], label = 'train')
plt.plot(hist.history['val_loss'], label = 'val')
plt.title('Leaves model :  Training & Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

###############################################################################



##############################Prediction#######################################
test_labels=test_batches.classes
print(test_labels)

predictions=model.predict(x=test_batches,steps=len(test_batches),verbose=0)

y_pred=predictions.argmax(axis=1)
print(y_pred)
cm=confusion_matrix(y_true=test_labels,y_pred=predictions.argmax(axis=1))


##############################################################################


###################################Confusion Matrix############################




# def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    
#     """This function prints and plots the confusion matrix
#     Normalization can be applied by setting normalization=True"""
    
#     plt.imshow(cm,interpolation="nearest",cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks=np.arange(len(classes))
#     plt.xticks(tick_marks,classes,rotation=45)
#     plt.yticks(tick_marks,classes)
    
    
#     if normalize:
#         cm=cm.astype('float')/cm.sum(axis=1)[:np.newaxis]
#         print("Normalized Confusion Matrix")
        
#     else:
#         print("Confusion Matrix without Normalization ")
        
#     print(cm)
    
#     thresh=cm.max()/2.
    
#     for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
#         plt.text(j,i,cm[i,j],horizontalalignment="center",
#         color="white" if cm[i,j] >thresh else "black")
    
#     plt.tight_layout()
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
    

test_batches.class_indices


cm_plot_labels=['Ak','Ala_Idris','Buzgulu','Dimnit','Nazli']
#plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title='Confusion Matrix')




# Confusion Matrix  & Pres  & Recall   & F1-Score


from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

print('Confusion Matrix')
print(confusion_matrix(test_labels, y_pred))

print('classification_Report')
print(classification_report(test_labels, y_pred, target_names=cm_plot_labels))

ax = plt.subplots(figsize=(5, 5))

sns.heatmap(cm, xticklabels=cm_plot_labels, yticklabels=cm_plot_labels, center = True, annot=True)

# disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=target_names)
# disp = disp.plot(cmap=plt.cm.Blues,values_format = 'g')
plt.title('Confusion Matrix')
plt.yticks(rotation=90,fontsize=10)
plt.xticks(fontsize=10)
#plt.ax.tick_params(axis='both', which='major', labelsize=10)

plt.show()

        
                
    
    















