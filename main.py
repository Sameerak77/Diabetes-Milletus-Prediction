import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("home.html", values=None)


@app.route("/predict", methods=['POST'])
def predict():
    preg = request.form.get('preg')
    gluc = request.form.get('gluc')
    bp = request.form.get('bp')
    st = request.form.get('st')
    ins = request.form.get('ins')
    bmi = request.form.get('bmi')
    dpf = request.form.get('dpf')
    age = request.form.get('age')
    val_list = [int(preg), int(gluc), int(bp), int(
        st), int(ins), bmi, dpf, int(age)]
    val_list = [np.array(val_list)]
    prediction = model.predict(val_list)
    output = round(prediction[0], 2)
    print(output)
    return render_template("home.html", values=output)
