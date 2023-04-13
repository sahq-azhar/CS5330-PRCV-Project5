#Syed Azhar Hussain Quadri
# CS 5330 Project 5

import cv2
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0)

# load the model
loaded_model = tf.keras.models.load_model("my_model.h5")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dsize = (28, 28)
    img = cv2.resize(gray, dsize)
    # required pre-processing to the image to be able to send this img to the model
    img_ = np.array(img).reshape((1, 28, 28, 1)).astype("float32") / 255
    # Display the resulting frame
    
    print(loaded_model.predict(img_)[0])
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

