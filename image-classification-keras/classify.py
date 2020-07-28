# USAGE
# python test_network.py --model Elephant_unknown.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()'''
camera = cv2.VideoCapture(0)
while(True):

    i=0
    while(True):
        ret,frame = camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(str(i)+'.png', frame)
            path=str(i)+'.png'
            image = cv2.imread(path)
            #image = cv2.imread(path)


            output = image.copy()



            # pre-process the image for classification
            image = cv2.resize(image, (28, 28))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # load the trained convolutional neural network
            print("[INFO] loading network...")
            model = load_model("Elephant_unknown.model")

            # classify the input image
            (unknown, Elephant) = model.predict(image)[0]

            # build the label
            label = "Elephant" if Elephant > unknown else "unknown"
            proba = Elephant if  Elephant > unknown else unknown
            label = "{}: {:.2f}%".format(label, proba * 100)

            # draw the label on the image
            output = imutils.resize(output , width=400)
            cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

            # show the output image
            cv2.imshow("Output", output)
            cv2.waitKey(0)
camera.release()
cv2.destroyAllWindows()
