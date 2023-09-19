from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/'
model = pickle.load(open("spam_ham_classifier.pkl", "rb"))
feature_extraction = pickle.load(open("feature_extractor.pkl", "rb"))

@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'spam_ham.jpg')
    return render_template("index.html", spam_ham_pic=pic1)

@app.route("/display" , methods=['GET', 'POST'])
def uploader():    
    if request.method=='POST':
        text_of_email = request.form["text_of_email"]
        input_mail = [text_of_email]
        input_data_features = feature_extraction.transform(input_mail)
        prediction = model.predict(input_data_features)
        if (prediction[0]==1):
            result="Ham"
        else:
            result="Spam"
        pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'spam_ham.jpg')
        return render_template("display.html", result=result, text_of_email=text_of_email, spam_ham_pic=pic1)

input_mail = ["Congrats, you have won a lottery for free"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)
if (prediction[0]==1):
    result="Ham"
else:
    result="Spam"

if __name__ == '__main__':
    app.run(debug=True) 