from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load("score.pkl")

print(model)

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    arr = np.asarray(data1, dtype='float64')
    arr = arr.reshape(1, -1)
    pred = int(round(model.predict(arr)[0][0]))
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)