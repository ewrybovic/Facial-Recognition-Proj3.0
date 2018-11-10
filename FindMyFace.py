import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as K

image_size = 96
image_paddiding = 50


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#set the format of keras so a 100 x 100 RGB image has shape (3, 100, 100)
K.set_image_data_format("channels_first")

# initialize the model
FRmodel = faceRecoModel(input_shape=(3,image_size,image_size))

# triplet loss is a method to calculate loss
# it minimizes the distance between an anchor and a positive(image that contains the same identity)
# and maximizes the distance between the anchor and a negative image(different identity)
# alpha is used to make sure the function does not try to optimize towrds 0
def triplet_loss(y_true, y_pred, alpha = 0.3):
	anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

	#reduce_sum gets the sum of the given axis
	# this is essentially pithag but with arrays thats why we use reduce_sum
	positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)

	negative_distance = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

	# the loss is the distance between the two images, but we add the alpha so the loss !=0
	basic_loss = tf.add(tf.subtract(positive_distance, negative_distance), alpha)

	# gets the max of the array and do a reduce_sum
	loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

	return loss

# compile the model using adam optimizer to minimize the loss, our loss method, and metrics of accuracy
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])

load_weights_from_FaceNet(FRmodel)

FRmodel.save("my_model.h5")

def prepare_database():
	database = {}

	# get every image in the directory
	for file in glob.glob("images/*"):
		# split the image and get the name of the person
		identity = os.path.splitext(os.path.basename(file))[0]
		# from fr_utils.py
		database[identity] = img_path_to_encoding(file, FRmodel)

	return database

def who_is_it(image, database, model):
	# from fr_utils.py
	encoding = img_to_encoding(image, model)

	min_distance = 100
	identity = None

	#Loop over the dictionary of names and encodings
	for (name, enc) in database.items():
		dist = np.linalg.norm(enc - encoding)
		print("distance for %s is %s" %(name, dist))

		# change to what I want
		if dist > 0.8:
			continue
		else:
			return name

def find_identity(frame, x1, y1, x2, y2):
	height, width, channels = frame.shape
	part_img = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
	return who_is_it(part_img, database, FRmodel)

cap = cv2.VideoCapture(0)

database = prepare_database()

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
	for (x,y,w,h) in faces:
		x1 = x - image_paddiding
		y1 = y - image_paddiding
		x2 = x + w + image_paddiding
		y2 = y + h + image_paddiding

		# predict who it is
		identity = find_identity(frame, x1, y1, x2, y2)
		print(identity)
		if identity is not None:
			font = cv2.FONT_HERSHEY_SIMPLEX
			# name will be the label at the value of id
			name = identity
			color = (0,0,0)
			stroke = 2
			cv2.putText(frame, name, (x1-5, y1), font, 1, color, stroke, cv2.LINE_AA)

		color = (255, 0, 0) #BGR 0-255 
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, stroke)

	cv2.imshow("Face Recog", frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
