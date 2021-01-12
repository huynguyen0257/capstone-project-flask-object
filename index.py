# from pathlib import Path
from flask_cors import CORS, cross_origin
from flask import Flask
from PIL import Image, ImageDraw
import io
import time
import base64
import numpy as np
import cv2
from flask import Flask, send_file, request, jsonify, redirect, Blueprint
import sys
from absl.flags import FLAGS
from absl import app, flags, logging
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)


sys.path.append('services/')
sys.path.append('route/')
from detectvideo import detectObjectV4


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# test code here
@app.route('/')
def index():
    return "Hello, World!"


@app.route('/detectobject', methods=['POST'])
def detectObject():
    start = time.time()
    data = request.get_json()
    if data is None:
        print("No valid request body, json missing!")
        return jsonify({'error': 'No valid request body, json missing!'})
    else:
        img_data = data['image']
        print("time parse: ", time.time() - start)
        # if time.time() - start > 0.5:
        #     print(img_data)
        start = time.time()
        decoded_data = base64.b64decode(img_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        result = detectObjectV4(img)
        print('result: ', result)
        print("time object: ", time.time() - start)
        

    return jsonify(result)



if __name__ == '__main__':
    app.run(use_reloader=False, debug=True, host= '0.0.0.0', port=5002)
    
