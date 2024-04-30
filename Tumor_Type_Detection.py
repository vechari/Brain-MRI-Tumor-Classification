#for downloading and accessing dataset
import os
import cv2

#for tensorflow classification and image printing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator #for data augmentation

#for GRADCAM
from gradcam import GradCAM

#Functions
"""Function: get_path
  gets path to directory based on current working directory
  used to access pictures"""
def get_path(directory):
    # Get the current working directory
    cwd = os.getcwd()
    # Path to the directory containing your images (assuming it's named "images")
    data_dir = os.path.join(cwd, directory)
    return data_dir

"""Function: set_datasets
  creates the training and testing sets based on passsed in parameters
  parameters:
    validation_split: validation split of dataset 
    data_dir: working directory for images in dataset
    img_height, img_width: height and width of images for normalization
    batch_size: number of photos from dataset per batch
    class_names: list of classs names"""
def set_datasets(validation_split, data_dir, img_height, img_width, batch_size, class_names):
   #set up training dataset with validation split (80% of images for tr-aining, and 20% of images for validation)
  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    class_names=class_names)

  #Set up training set
  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    class_names=class_names)
  
  return train_ds, val_ds

"""Function: visualize dataset'
  prints first 10 images from datset for viewing"""
def visualize_dataset():
   #visualize data
  #first 10 images of training data set
  plt.figure(figsize=(10, 10))
  for images, labels in train_ds.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")

"""Function: build_model
  create and compile sequential model"""
def build_model():
  #create Keras Model
  num_classes = len(class_names)

  model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])
  #compile the model
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  #model summary
  model.summary()

  return model

"""Function: train_model
  train the model for a specified number of epochs
  visualize training metrics (accuracy and loss)"""
def train_model(model, train_ds, val_ds, epochs):
  #train the model
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  #visualize training metrics
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.xlabel('Epoch Trial')
  plt.ylabel('Accuracy')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.xlabel('Epoch Trial')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')

  plt.savefig('metrics_type.png')
  plt.show()

"""Function: get_image
  get file path and image to specific image for prediction"""
def get_image(path, img_num):
  pred_file_path = os.path.join(path, image_files[img_num]) #index is which image you are going to use for prediction
  img = mpimg.imread(pred_file_path)
  
  # Display the image
  plt.imshow(img)
  plt.axis('off')  # Turn off axis
  plt.show()

  return pred_file_path, img


"""Function: make_prediction
  make prediction based on trained model for novel image"""
def make_prediction(pred_file_path, img_height, img_width):
   #predict if there is a tumor
  img = tf.keras.utils.load_img(
      pred_file_path, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  img_array = data_augmentation(img_array)

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  return score #return score and augmented image for gradcam


#Main Script

#Training
images_path = 'tumor_type_images'
data_dir = get_path(images_path) #get path to images

#print how many folders in each subfolder
for root, dirs, files in os.walk(data_dir):
        file_count = len(files)
        print(f"Folder: {root}, Files: {file_count}")

#set parameters for the loader
batch_size = 32
img_height = 180
img_width = 180
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
validation_split = 0.2

train_ds, val_ds = set_datasets(validation_split, data_dir, img_height, img_width, batch_size, class_names) #create training and testing datasets
#testing 2 classes: yes and no
class_names = train_ds.class_names #confirm class names
print(class_names)

# Define data augmentation parameters
data_augmentation = Sequential([
    layers.RandomRotation(0.1),        # Randomly rotate images by up to 10 degrees
    layers.RandomZoom(0.1),            # Randomly zoom images by up to 10%
])


model = build_model() #build sequential model

AUTOTUNE = tf.data.AUTOTUNE #configure data to for model performance

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #configure training dataset for training
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) #configure validation dataset for training

train_model(model, train_ds, val_ds, epochs=25) #train model for specified number of epochs


#Prediction on novel Image: 
  #Classify new image (not in dataset) as "no" tumor or "yes" tumor
pred_path = 'pred'
pred_dir = get_path(pred_path) #get path to prediction file

# extract image files
image_files = os.listdir(pred_dir)
print(image_files)
img_nums = [0,1,2,3] #1: no, 0: glioma, 2: meningoma, 3: pituitary
for img_num in img_nums:
  pred_file_path, img= get_image(pred_path, img_num) #get file path and image for prediciton

  score= make_prediction(pred_file_path, img_height, img_width) #make prediction
  print(score)
  outcome = np.argmax(score)  #highest score is the most probable classification

  #print prediction
  if class_names[outcome] == 'no':
    print("This image does not contains a tumor with a {:.2f} percent confidence."
      .format( 100 * np.max(score)))
  elif class_names[outcome] == 'meningioma_tumor':
    print("This image contains a meningioma tumor with a {:.2f} percent confidence."
      .format( 100 * np.max(score)))
  elif class_names[outcome] == 'glioma_tumor':
    print("This image contains a glioma tumor with a {:.2f} percent confidence."
      .format( 100 * np.max(score)))
  else:
      print(
      "This image contains a pituitary tumor with a {:.2f} percent confidence."
      .format(100 * np.max(score))
      )

  #create GRADCAM of prediction
  #convert model into a functional model 
  inputs = model.inputs
  outputs = model.layers[-1].output
  functional_model = Model(inputs=inputs, outputs=outputs)

  gradcam = GradCAM(functional_model, outcome, 'conv2d_5') #create gradcam model

  resized_image = cv2.resize(img, (180, 180))


  input_image = np.expand_dims(resized_image, axis=0) # Add the batch size dimension to compute heatmap
  heatmap = gradcam.compute_heatmap(input_image) #compute heatmap based on gradcam class
  heatmap = cv2.resize(heatmap, (180,180)) #resize heatmap to print out

  image = cv2.resize(img, (img_height, img_width)) #resize image to overlay with heatmap
  print(heatmap.shape, image.shape)

  (heatmap, output) = gradcam.overlay_heatmap(heatmap, image, alpha=0.5) #overlay heatmap and original image

  #Visualize Results
  # Display heatmap
  plt.imshow(heatmap, cmap='jet')
  plt.colorbar()
  plt.title('Heatmap')
  plt.savefig(f'heatmap{img_num}.png')
  plt.show()

  # Display overlaid image
  plt.imshow(output)
  plt.title('Overlaid Image')
  plt.savefig(f'overlaid{img_num}.png')
  plt.show()