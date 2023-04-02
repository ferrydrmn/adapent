import os
import base64
import imghdr
import shortuuid
import numpy as np
import tensorflow as tf
from flask import request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow import device
from tensorflow.keras import models, losses
from tensorflow.keras.optimizers import Adam

from script import app
from script.model import googlenet


with device('/CPU:0'):
    model = models.load_model(os.path.join(app.root_path, 'model.h5'))


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    # imgEncoded = request.form['image']
    # imgDecoded = base64.b64decode(imgEncoded)
    # imgExtension = imghdr.what(None, h=imgDecoded)

    # if imgDecoded and imgExtension not in ['png', 'jpg', 'jpeg']:
    #     return jsonify({'message': 'Ekstensi gambar harus JPG, JPEG, atau PNG!'})
    
    # filename = f'{shortuuid.uuid()}.{imgExtension}'
    
    # with open(f'{app.root_path}/static/uploads/{filename}', 'wb') as fh:
    #     fh.write(imgDecoded)

    # description = '''
    #     Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    #     Ut tristique eleifend viverra. Sed commodo varius nisi sagittis sagittis. 
    #     Proin hendrerit mauris vitae massa commodo vestibulum. 
    #     Pellentesque et nulla ut mauris ultrices fermentum eu a purus. 
    #     Cras tincidunt vel enim vitae ullamcorper. 
    #     Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    #     Cras maximus mi non felis pulvinar, id imperdiet lectus bibendum. 
    #     Proin a varius sapien. Phasellus eu magna lectus. 
    #     Nam sit amet accumsan ex. 
    #     Mauris quis mauris dignissim, tristique turpis a, efficitur elit. 
    #     Quisque maximus sem a metus ullamcorper, ac luctus enim sodales. 
    #     Suspendisse ornare turpis ac augue dictum posuere. 
    #     Cras viverra, nunc ac pulvinar blandit, neque elit vestibulum velit, a aliquam neque ipsum non libero. 
    #     Aenean ultrices, nibh id blandit convallis, metus felis aliquet sem, quis maximus nunc tellus vel diam. 
    #     Ut posuere pharetra ultricies.
    # '''

    imgExtension = 'jpeg'
    filename = 'HeDT7bpgwjTNSAjEA78wc5.jpeg'
    path_to_file = url_for('static', filename=f'uploads/{filename}').replace('/', '\\')

    image = tf.io.read_file(app.root_path.replace('/', '\\') + path_to_file)
    
    if imgExtension in ['jpeg', 'jpg']:
        image = tf.image.decode_jpeg(image, channels=3)
    elif imgExtension == 'png':
        image = tf.image.decode_png(image, channels=3)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = np.expand_dims(image, axis=0)
    
    with device('/CPU:0'):
        predictions = model.predict(image)
        predictions = np.argmax(predictions[0])

    return jsonify({'message': 'Proses prediksi berhasil dilakukan!', 'result': str(predictions), 'description': 'ABC'})

@app.route('/upload', methods=['POST'])
def upload():
    
    data = request.get_json()

    if not data['file']:
        return jsonify({'message': 'Null!'})

    lenmax = len(data['file']) - len(data['file'])%4
    
    file = base64.b64decode(data['file']).decode('utf-8')

    ext = imghdr.what(None, h=file)

    if file and ext in ['png', 'jpg', 'jpeg']:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'gambar.{ext}'))
        return jsonify({'message': 'Prediksi berhasil dilakukan!'})
    else:
        return jsonify({'message': 'Esktensi gambar yang diperbolehkan adalah PNG, JPG, atau JPEG!'})
    
@app.route('/test')
def test():
    return jsonify({'message': 'Halo Ferry!'}), 200

@app.route('/post', methods=['POST'])
def post():
    value = request.form['value']
    return jsonify({'message': value}), 200
    