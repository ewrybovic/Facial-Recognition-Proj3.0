#!/usr/bin/env python
from flask import Flask, render_template, Response, request, redirect
import cv2
import sys
import numpy
import datetime
import time
import os
import glob
import json
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as K 

image_size = 96

# The amount of time to wait when a face is found before taking a picture
wait_time = 5

# OpenCV gets the face but FaceNet needs the whole head
image_padding = 30
image_x = 640
image_y = 480

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'static')
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

# Face classifier from OpenCV
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#face_model = load_model('my_model.h5', custom_objects={'triplet_loss': triplet_loss})

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

def prepare_database(model):
	database = {}

	for file in glob.glob("static/images/*"):
		identity = os.path.splitext(os.path.basename(file))[0]
		database[identity] = img_path_to_encoding(file, model)
	return database

def who_is_it(image, model):
	encoding = img_to_encoding(image, model)

	min_distance = 100
	identity = None

	for (name, enc) in prepare_database(model).items():
		dist = np.linalg.norm(enc - encoding)
		print("Distance for %s is %s" %(name, dist))

		#THIS DECIDES IF WE KNOW A FACE ORE NOT
		#CHANGE THIS TO GET DIFFERENT RESULTS
		if dist > 0.8:
			continue
		else:
			print("Identity is %s" %(name))
			return name

def find_identity(frame, x1, y1, x2, y2, model):
	height, width, channels = frame.shape
	part_img = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

	return who_is_it(part_img, model)


# Make the flask app
app = Flask(__name__)

# The starting index of our webpage
@app.route('/', methods=['GET'])
def index():
	return render_template('index.html', message = "Video Streaming Demonstration")

# If the user clicks the results button we check if there is a name in the text file
# then redirect
@app.route('/', methods=['POST'])
def results_page():
	file = open("name.txt", "r")
	name = file.read()
	if name == "none":
		return not_found()
	else:
		return found()


# Function for getting the frame from webcam using opencv
def get_frame():
	# Initialize variables
	image_zoom_x = 640
	image_zoom_y = 480
	did_take_pic = False
	timeFoundFace = 0
	picture = None
	timeForNextStep = time.time() + wait_time
	x1, y1, x2, y2 = 0, 0, 0, 0
	finishedProcessing = False
	image_padding = 30
	image_x = 1280
	image_y = 720
	face_perimeter = 0
	too_close = False

	# Set the loading frame for the user
	loading_frame = np.zeros((image_y, image_x, 3), np.uint8)
	text = "Please wait while the FaceNet model loads"
	location = (int(image_x/4),int(image_y/2) - 20)
	font = cv2.FONT_HERSHEY_SIMPLEX
	color = (255,255,255)
	stroke = 3
	size = 1
	cv2.putText(loading_frame, text, location, font , size, color, stroke, cv2.LINE_AA)
	loading_frame = cv2.imencode('.jpg', loading_frame)[1]
	loading_frame = loading_frame.tostring()
	yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+loading_frame+b'\r\n')

	# Compile the FaceNet model
	K.set_image_data_format("channels_first")
	FRmodel = faceRecoModel(input_shape=(3, image_size, image_size))
	FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
	load_weights_from_FaceNet(FRmodel)

	#make the default identity to be none on starup
	save_identity("none")

	# Make the video catpture
	camera=cv2.VideoCapture(0)
	template = 'picture'
	# Constant loop for getting the image
	while True:
		# IF we already took a picture of the face
		# inform the user to click the button
		if did_take_pic:
			inform_user_frame = np.zeros((image_y, image_x, 3), np.uint8)
			text = "Please click the button underneath this frame"
			location = (int(image_x/4), int(image_y/2) - 20)
			color = (255,255,255)
			stroke = 3
			size = 1 
			cv2.putText(inform_user_frame, text, location, font, size, color, stroke, cv2.LINE_AA)
			inform_user_frame = cv2.imencode('.jpg', inform_user_frame)[1]
			inform_user_frame = inform_user_frame.tostring()
			yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+inform_user_frame+b'\r\n')
		else:	

			found_Face = False
			# get an image from the video capture
			retval, im = camera.read()
			im = cv2.resize(im, (image_x, image_y))
			# change the image into a grey image
			gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			# get an array of face locations that is returned from the cascade
			faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

			# for every face we have we draw a rectangle around it
			for (x, y, w, h) in faces:
				found_Face = True
				x1 = x - image_padding
				y1 = y - image_padding
				x2 = x + w + image_padding
				y2 = y + h + image_padding
				face_perimeter = ((x2-x1)*2)+((y2-y1)*2)
				text_x = x1
				text_y = y1 - 10
				cv2.rectangle(im, (x1,y1), (x2, y2), (255,0,0), 1)
				break
			
			if face_perimeter >= 1900:
				too_close = True
			else:
				too_close = False

			# Get the text to print on the image
			if found_Face and not did_take_pic:
				timeFoundFace = time.time()
				stroke = 1
				image_zoom_x -= 10
				image_zoom_y -= 10
				if timeFoundFace >= timeForNextStep:
					if finishedProcessing:
						identity = find_identity(im, x1, y1, x2, y2, FRmodel)
						did_take_pic = True
						if identity is not None:
							save_identity(identity)
						else:
							save_identity("none")

						del(camera)
					else:
						text =  "Processing"
						save_Picture(im[y1+2:y2-2, x1+2:x2-2])
						finishedProcessing = True
				else:
					time_left = int(timeForNextStep - timeFoundFace)
					if time_left == 0:
						text = "Taking Pic"
					else:
						if too_close:
							text = "Move back please"
							timeForNextStep = time.time() + wait_time
						else:
							text = "Waiting for " + str(time_left) + " sec"
			else:
				timeFoundFace = 0
				image_zoom_x = 640
				image_zoom_y = 480
				timeForNextStep = time.time() + wait_time
				text_x = 25
				text_y = 35
				stroke = 2
				finishedProcessing = False
				text =  "No Face Found"


			# Write the time to the image wil use this as a way to write instructions
			font = cv2.FONT_HERSHEY_SIMPLEX
			color = (0,0,0)
			location = (text_x, text_y)
			cv2.putText(im, text, location, font, 1, color, stroke, cv2.LINE_AA)


			imgencode=cv2.imencode('.jpg',im)[1]
			# then convert it to a string of values
			stringData=imgencode.tostring()
			# return the value. yield means it will return the value but keep running the code
			yield (b'--frame\r\n'
				b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

@app.route('/calc')
def calc():
	# get a response for the img in the index.html
	# get_frame will constantly return a string of values
	 return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

def contact():
	if "response_page" in request.form:
		print("Button pressed")
		return render_template("Found.html")
	else:
		index()

def save_identity(identity):
	file = open("name.txt", "w+")
	file.write(identity)

@app.route('/found')
def found():
    file = open("name.txt", "r")
    name = file.read()
    with open('users.json') as data:
        json_Data = json.load(data)
    return render_template('Found.html', data=json_Data, name=name)

@app.route('/not_found')
def not_found():
	return render_template('Not_Found.html')

@app.route('/updated', methods = ['POST'])
def updated():
    file = open("name.txt", "r")
    name = file.read()
    if name == "none":
        firstName = request.form['firstname']
        lastName = request.form['lastname']
        favorites = request.form['favorites']
        new_user = {
            "name": firstName + ' ' + lastName,
            "fName": firstName,
            "lName": lastName,
            "favorites": [favorites]
        }
        with open('users.json') as data:
            json_Data = json.load(data)
            json_Data.append(new_user)
    else:
        new_Favorite = request.form['favorites']
        with open('users.json') as data:
            json_Data = json.load(data)
            for i in json_Data:
                if i["fName"] == name:
                    i["favorites"].append(new_Favorite)
    with open('users.json', 'w') as outFile:
        json.dump(json_Data, outFile)
    return render_template('Found.html', data=json_Data, name=name)

def save_Picture(image):
	filename = "test.jpg"
	cv2.imwrite(os.path.join(final_directory , filename), img=image)

if __name__ == '__main__':
	app.run(host='localhost', debug=True, threaded=True)
