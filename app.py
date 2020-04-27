import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_iris.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 0:
        res_val = "** Iris-Setosa **"
    elif output==1:
        res_val = "** Iris-Versicolour **"
    else: 
        res_val = "** Iris-Virginica **"

    return render_template('index.html', prediction_text='Flower is :: {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=False)
