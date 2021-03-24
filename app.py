import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    val_req = list((request.form.values()))
    classifier = [int(x) for x in request.form.values()][0]
    int_features = [int(x) for x in request.form.values()][1:]
    final_features = [np.array(int_features)]
    if classifier == 1:
        prediction = model.predict(final_features)
    elif classifier == 2:
        prediction = model1.predict(final_features)
    else:
        prediction = model2.predict(final_features)

    if prediction == 0:
        return render_template ('index.html',prediction_text='Eligibility criteria not satisfied, hence no Promotion',val_req=val_req)

    else:
        return render_template ('index.html',prediction_text='The Employee is eligible for Promotion')



if __name__ == "__main__":
    app.debug=True
    app.run(debug=True)