import sys
from tensorflow import keras
import cv2
import numpy as np
import csv

CLASS_NAME = {}
WAIT_KEY = 10000

def main():

    # Create label
    csv_file = "traffic.csv"
    with open(csv_file, encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            CLASS_NAME[int(row["number"])] = row["sign"]

    # Check valid argv
    if len(sys.argv) != 3:
        print("Usage: python3 recognition.py model image")

    # Create model
    model = keras.models.load_model(sys.argv[1])

    # Open image
    image = cv2.imread(sys.argv[2])
    
    # Resize image
    resize_image = cv2.resize(image,(30,30))

    # Expand dimensions
    input_image = np.expand_dims(resize_image, axis=0)

    # Prediction
    classification = model.predict(input_image).argmax()

    # Show the result
    if classification is not None:

        # Create a white background image
        white_image = np.ones((1000,1000,3), dtype=np.uint8)*255

        # Resize image
        resize_image = cv2.resize(image,(350,350))

        # Calculate the position to place the available image on the white image
        x_offset = int((1000 - resize_image.shape[1])/2)
        y_offset = int((1000 - resize_image.shape[0])/2)

        # Overlay the available image onto the white image
        white_image[y_offset: y_offset + resize_image.shape[0], x_offset: x_offset + resize_image.shape[1]] = resize_image

        # Write predict
        text = "Predicted class: "+ CLASS_NAME[int(classification)]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
        text_x = int((white_image.shape[1] - text_size[0]) / 2)

        # Display the result
        cv2.putText(white_image, text, (text_x,750), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,0), 1, cv2.LINE_AA)
        cv2.imshow('Overlayed Image', white_image)
        cv2.waitKey(WAIT_KEY)
        cv2.destroyAllWindows()   
       
    else:
        print("Failed to predict class.")



if __name__ == "__main__":
    main()