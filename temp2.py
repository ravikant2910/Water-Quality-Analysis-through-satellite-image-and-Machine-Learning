import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import rasterio
import os

# Define the path to your training data directory
data_dir = "E:\RBL-Water Project\archive"

# Specify the subdirectories for positive (water) and negative (non-water) classes
positive_class_dir =  "E:\RBL-Water Project\archive\X_train.tif"
negative_class_dir = "E:\RBL-Water Project\archive\y_train.tif"

# Function to load and preprocess TIFF images using rasterio
def load_tiff_images(directory):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            with rasterio.open(os.path.join(directory, filename)) as src:
                img = src.read()  # Read the image using rasterio
            img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)
            image_list.append(img)
    return np.array(image_list)

# Load positive (water) and negative (non-water) images
positive_images = load_tiff_images(positive_class_dir)
negative_images = load_tiff_images(negative_class_dir)

# Check the shape of loaded images
print("Positive Images Shape:", positive_images.shape)
print("Negative Images Shape:", negative_images.shape)

# Check if the images need resizing to a consistent shape
new_height = 128
new_width = 128

# Resize each image to the new dimensions if needed
positive_images_resized = [np.resize(img, (new_height, new_width, 4)) for img in positive_images]
negative_images_resized = [np.resize(img, (new_height, new_width, 4)) for img in negative_images]

# Convert the lists back to numpy arrays
positive_images = np.array(positive_images_resized)
negative_images = np.array(negative_images_resized)

# Check the shape after resizing
print("Positive Images Resized Shape:", positive_images.shape)
print("Negative Images Resized Shape:", negative_images.shape)

# Rest of the code remains the same as in the previous response...

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 4)))  # Assuming input images are 128x128 pixels with 4 channels
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation to improve model performance
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

# Train the model
batch_size = 32
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy*100:.2f}%")
