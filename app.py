import pickle
from flask import Flask, request, render_template, url_for, redirect

app = Flask(__name__)
classifier = pickle.load(open('model.pkl', 'rb'))[0]
scaler = pickle.load(open('model.pkl', 'rb'))[1]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if(request.method == 'POST'):
        variance = request.form['variance']
        skewness = request.form['skewness']
        curtosis = request.form['curtosis']
        entropy = request.form['entropy']
        prediction = classifier.predict(scaler.transform([[variance, skewness, curtosis, entropy]]))[0]
        
        if(prediction):
            prediction_text = 'The Bank Note is a Fake!'
        else:
            prediction_text = 'The Bank Note is Authentic!'
        return render_template('index.html', prediction_text=prediction_text)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)