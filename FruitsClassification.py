import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPool2D,BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import cv2
import time
from sklearn.metrics import *
from sklearn.metrics import classification_report
np.random.seed(1)

target_shape = (200,200)
train_imgs = []
train_labels = []
train_file = "/content/drive/MyDrive/Colab Notebooks/NNExam/train_zip/train"

for file in os.listdir(train_file):
#     if the file have jpg its an image
    if file.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(train_file,file))
#         converting images to rgb 
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#       appending labels
        train_labels.append(file.split('_')[0])
#       resizing images
        img = cv2.resize(img,target_shape)
        train_imgs.append(img)

# converting labesls to one hot encoded  
train_labels = pd.get_dummies(train_labels).values
train_imgs = np.array(train_imgs)

# spliting data into train and valid sets
x_train,x_valid,y_train,y_valid = train_test_split(train_imgs,train_labels,random_state=1,test_size=0.2)


test_imgs = []
test_labels = []
test_file = "/content/drive/MyDrive/Colab Notebooks/NNExam/test_zip/test"

for file in os.listdir(test_file):
    if file.split(".")[1] == "jpg":
#       if the file have jpg its an image
        img = cv2.imread(os.path.join(test_file,file))
    #   converting images to rgb 
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # appending labels
        test_labels.append(file.split("_")[0])
        img = cv2.resize(img,target_shape)
        test_imgs.append(img)
        
# converting labesls to one hot encoded          
test_labels = pd.get_dummies(test_labels).values
test_imgs = np.array(test_imgs)



number_of_class=4

# bulding the model architecture
model = Sequential()

# Input Layer
model.add(Conv2D(kernel_size=(3,3),filters=16,activation='relu',input_shape=(200,200,3,)))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))

# 1st Hidden Layer
model.add(Conv2D(kernel_size=(3,3),filters=16,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))

# 2nd Hidden Layer
model.add(Conv2D(kernel_size=(3,3),filters=16,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))

# 3rd Hidden Layer
model.add(Conv2D(kernel_size=(3,3),filters=16,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))

# 4th Hidden Layer
model.add(Conv2D(kernel_size=(3,3),filters=16,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))

# 5th Hidden Layer
model.add(Conv2D(kernel_size=(3,3),filters=16,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))

# 6th Hidden Layer
model.add(Conv2D(kernel_size=(5,5),filters=16,activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

# 2nd Fully Connected Layer
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

# Final Output Layer
model.add(Dense(number_of_class,activation='softmax'))

model.summary()


# compiling the model using adam optimizer and cros entropy loss
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])

# training the model
epochs = 40
batch_size = 16

start = time.time()

trained_model = model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_valid,y_valid))

total_time = time.time() - start

print("Model trained in {:.4f} sec".format(total_time))


# plotting the loss and accuracy curve for each phase
train_loss = trained_model.history['loss']
val_loss = trained_model.history['val_loss']
train_acc = trained_model.history['accuracy']
val_acc = trained_model.history['val_accuracy']

epochs_range = range(epochs)

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.plot(epochs_range,train_loss,label="train_loss")
plt.plot(epochs_range,val_loss,label="val_loss")
plt.legend(loc=0)
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(epochs_range,train_acc,label="train_acc")
plt.plot(epochs_range,val_acc,label="val_acc")
plt.legend(loc=0)
plt.title("Accuracy")

plt.show()


# testing the model on the test set
model.evaluate(test_imgs,test_labels)

# test item prediction
testLabelPredicted = model.predict(test_imgs)
#print (testLabelPredicted)
#testLabelPredicted = testLabelPredicted.argmax(axis=-1)
testLabelPredicted=np.argmax(testLabelPredicted, axis=-1)
testLabelGold = np.argmax(test_labels, axis=-1)
#print (testLabelGold)

# Evaluation
results = confusion_matrix(testLabelGold, testLabelPredicted) 
    
print ('Confusion Matrix :')
print (results) 
target_names = ['Apple','Banana','Mixed','Orange']
print(classification_report(testLabelGold, testLabelPredicted, target_names=target_names))

print ('Recall Score :',recall_score(testLabelGold, testLabelPredicted, average='micro'))
print ('Precision Score :',precision_score(testLabelGold, testLabelPredicted, average='micro'))
print ('F1 Score :',f1_score(testLabelGold, testLabelPredicted, average='micro'))
print ('Accuracy :',accuracy_score(testLabelGold, testLabelPredicted))
