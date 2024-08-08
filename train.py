import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import rasterio
from skimage.transform import resize  # Import the resize function

.

# Define the paths to your training data directories
data_dir = "E:/RBL-Water Project/archive/X_train.tif"
label_dir = "E:/RBL-Water Project/archive/y_train.tif"

# Load positive (water) and negative (non-water) images using rasterio
def load_tiff_images(directory):
    with rasterio.open(directory) as src:
        img = src.read()  # Read the image using rasterio
    return img

# Load positive (water) and negative (non-water) images
positive_images = load_tiff_images(data_dir)
negative_images = load_tiff_images(label_dir)

# Transpose images to (height, width, channels)
positive_images = np.transpose(positive_images, (1, 2, 0))
negative_images = np.transpose(negative_images, (1, 2, 0))

# Extract patches from the images
def extract_patches(image, patch_size, stride):
    patches = []
    height, width, _ = image.shape
    for y in range(0, height - patch_size[0] + 1, stride[0]):
        for x in range(0, width - patch_size[1] + 1, stride[1]):
            patch = image[y:y + patch_size[0], x:x + patch_size[1], :]
            patches.append(patch)
    return patches

patch_size = (128, 128)
stride = (64, 64)

positive_patches = extract_patches(positive_images, patch_size, stride)
negative_patches = extract_patches(negative_images, patch_size, stride)

# Ensure the number of positive patches matches the number of negative patches
min_num_patches = min(len(positive_patches), len(negative_patches))
positive_patches = positive_patches[:min_num_patches]
negative_patches = negative_patches[:min_num_patches]

# Resize patches to a consistent shape (128, 128, 3)
new_height, new_width = patch_size
positive_patches_resized = [resize(patch, (new_height, new_width, 3), anti_aliasing=True) for patch in positive_patches]
negative_patches_resized = [resize(patch, (new_height, new_width, 3), anti_aliasing=True) for patch in negative_patches]

# Stack the patches to create training data
positive_patches_stacked = np.stack(positive_patches_resized, axis=0)
negative_patches_stacked = np.stack(negative_patches_resized, axis=0)

# Create labels (1 for water, 0 for non-water)
positive_labels = np.ones(len(positive_patches_stacked))
negative_labels = np.zeros(len(negative_patches_stacked))

# Combine patches and labels
patches = np.concatenate((positive_patches_stacked, negative_patches_stacked), axis=0)
labels = np.concatenate((positive_labels, negative_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(patches, labels, test_size=0.2, random_state=42)

# Normalize the pixel values to be in the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to categorical format
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))  # Assuming input images are 128x128 pixels with 3 channels (RGB)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D((2, 2))
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

# Save the trained model
#model.save("E:/RBL-Water Project/trained_model.h5")
