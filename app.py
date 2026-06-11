from flask import Flask, render_template, request, url_for, Markup, jsonify
import pickle
import pandas as pd
from random import randrange

app = Flask(__name__)
pickle_in = open('model_fakenews.pickle', 'rb')
pac = pickle.load(pickle_in)
tfid = open('tfid.pickle', 'rb')
tfidf_vectorizer = pickle.load(tfid)

# Load random news dataset
data = pd.read_csv("random_dataset.csv")  # Ensure you have this dataset

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/newscheck')
def newscheck():    
    abc = request.args.get('news')    
    input_data = [abc.rstrip()]
    # Transforming input
    tfidf_test = tfidf_vectorizer.transform(input_data)
    # Predicting the input
    y_pred = pac.predict(tfidf_test)
    return jsonify(result=y_pred[0])

@app.route('/generate_random_text')
def generate_random_text():
    index = randrange(0, len(data))  # Get a random index
    random_text = data.loc[index, 'text']  # Adjust column name based on your dataset
    return jsonify(random_text=random_text)

if __name__ == '__main__':
    app.run(debug=True)
