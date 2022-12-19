import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'models/XgboostClassifier.bin'


with open(model_file, 'rb') as f_in:
    dv, LE, model = pickle.load(f_in)

app = Flask('predict')


@app.route('/predict', methods=['POST'])
def predict():
    music = request.get_json()

    X = dv.transform([music])
    y_pred = model.predict(X)
    pred = LE.classes_[int(y_pred)]
    print(f'music genre is {pred}')
    return jsonify(pred)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='9696')
