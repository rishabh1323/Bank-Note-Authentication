import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
classifier = pickle.load(open('model.pkl', 'rb'))[0]
scaler = pickle.load(open('model.pkl', 'rb'))[1]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict(scaler.transform([[variance, skewness, curtosis, entropy]]))[0]
    return "Predicted value is " + str(prediction)


if __name__ == '__main__':
    app.run(debug=True)