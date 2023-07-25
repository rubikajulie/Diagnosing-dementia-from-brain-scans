import numpy as np
import os
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,f1_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import cv2
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = [229,229]

epochs = 500
batch_size = 64


dement_path = 'C:/Users/archa/Downloads/Alzheimer Detection new -11/Alzheimer Detection new/train/Demented'
nondement_path = 'C:/Users/archa/Downloads/Alzheimer Detection new -11/Alzheimer Detection new/train/NonDemented'

# Function to get a list of image files from a directory and its subdirectories
def get_image_files_from_directory(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
    return image_files

# Function to check if a file is an image based on its extension
def is_image_file(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # Add more if needed
    _, ext = os.path.splitext(file_path)
    return ext.lower() in valid_extensions

# Get a list of image files in the "Demented" directory and its subdirectories
Demfiles = get_image_files_from_directory(dement_path)

NonDemfiles = glob( nondement_path +'/*' )


print("First 5 NonDem Files: ",NonDemfiles[0:5])
print("Total Count: ",len(NonDemfiles))
print("First 5 MildDem Files: ",Demfiles[0:5])
print("Total Count: ",len(Demfiles))


Dem_labels = []
NonDem_labels = []

Dem_images=[]
NonDem_images=[]

for i in range(len(Demfiles)):
  image = cv2.imread(Demfiles[i]) 
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
  image = cv2.resize(image,(229,229)) 
  Dem_images.append(image) 
  Dem_labels.append('Demented') 
for i in range(len(NonDemfiles)):
  image = cv2.imread(NonDemfiles[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(229,229))
  NonDem_images.append(image)
  NonDem_labels.append('NonDemented')

#   def plot_images(images, title):
#     nrows, ncols = 5, 8
#     figsize = [10, 6]

#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=(1, 1, 1))

#     for i, axi in enumerate(ax.flat):
#         axi.imshow(images[i])
#         axi.set_axis_off()

#     plt.suptitle(title, fontsize=24)
#     plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
#     plt.show()
# plot_images(Dem_images, 'Demented Alzheimers Scan')
# plot_images(NonDem_images, 'NonDemented Alzheimers Scan')

Dem_images = np.array(Dem_images) / 255
NonDem_images = np.array(NonDem_images) / 255

Dem_x_train, Dem_x_test, Dem_y_train, Dem_y_test = train_test_split(
    Dem_images, Dem_labels, test_size=0.2)
NonDem_x_train, NonDem_x_test, NonDem_y_train, NonDem_y_test = train_test_split(
    NonDem_images, NonDem_labels, test_size=0.2)


X_train = np.concatenate((NonDem_x_train, Dem_x_train), axis=0)
X_test = np.concatenate((NonDem_x_test, Dem_x_test), axis=0)
y_train = np.concatenate((NonDem_y_train, Dem_y_train), axis=0)
y_test = np.concatenate((NonDem_y_test, Dem_y_test), axis=0)

y_train = LabelBinarizer().fit_transform(y_train)
y_train = to_categorical(y_train)

y_test = LabelBinarizer().fit_transform(y_test)
y_test = to_categorical(y_test)

# inception = InceptionV3(weights="imagenet", include_top=False,
#     input_tensor=Input(shape=(229, 229, 3)))

# outputs = inception.output
# outputs = Flatten(name="flatten")(outputs)
# outputs = Dropout(0.4)(outputs)
# outputs = Dense(2, activation="softmax")(outputs)

model = load_model('C:/Users/archa/Downloads/Alzheimer Detection new -11/Alzheimer Detection new/AD_Model_Inception_V3_Custom.h5')

y_pred = model.predict(X_test, batch_size=batch_size)

prediction=y_pred[1:10]
for index, probability in enumerate(prediction):
  if probability.item(0) > 0.5:
        plt.title('%.2f' % (probability.item(0)*100) + '% Demented')
  else:
        plt.title('%.2f' % ((1-probability.item(0))*100) + '% NonDemented')
  plt.style.reload_library
  plt.imshow(Dem_images[index])
  plt.show()