import numpy as np

from tensorflow import keras
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img

from flask import Flask, request, jsonify


#Load the model
best = 'CS107_0.995.h5'
model = keras.models.load_model(best)


app = Flask('pistachio-classifier')

@app.route('/classify', methods=['POST'])
def classify():
    data=request.get_json()
    path = get_file(origin=data.get('url'))
    img = load_img(path, target_size=(150,150))
    x = np.array(img)
    x= x/255
    x = np.array([x])
    
    pred = model.predict(x)
    
    results = dict(
        zip(
            ['Kirmizi_Pistachio', 'Siirt_Pistachio'],
            (round(float(pred[0][0]),2), round(float(pred[0][1]),2))))
    
    return jsonify(results)

if __name__ =='__main__':
    app.run(debug=True, host ='0.0.0.0', port=9696)