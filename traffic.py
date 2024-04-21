import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    Return tuple `(images, labels)`. 
    """
    
    # Create the list of label and image
    images = []
    labels = []

    # Get a list of directories in the root directory
    directories = [directory for directory in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, directory))]

    # Loop through each directory
    for directory in directories:
        
        # Create a label
        label = int(directory)

        # Get the full path of current directory
        current_directory = os.path.join(data_dir, directory)
        #print(current_directory)

        # Get a list of file in the current directory
        files = [file for file in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, file))]

        # Loop through file name
        for file in files:

            # Get the full path of the file
            current_file = os.path.join(current_directory,file)
            #print(current_file)

            # Read image from file
            image = cv2.imread(current_file)

            # Resize the image
            resize_image = cv2.resize(image,(IMG_WIDTH, IMG_HEIGHT))

            # Convert it to numpy array
            numpy_image = np.array(resize_image)
            #print(numpy_image)

            # Add image to the list
            images.append(numpy_image)

            # Add label to the list
            labels.append(label)
            
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # Apply Convolutional layer and Max-pooling layer for low-level features
        tf.keras.layers.Conv2D(
            30, (3, 3), activation="softmax", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

         # Apply Convolutional layer and Max-pooling layer for high-level features
        tf.keras.layers.Conv2D(
            40, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        

        # Flatten unit
        tf.keras.layers.Flatten(),

        # Add hidden layer with some units, with ReLU activation
        tf.keras.layers.Dense(215, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add output layer with 1 unit, with sigmod activation
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
