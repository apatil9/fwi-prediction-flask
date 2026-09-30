import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pf
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['POST','GET'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled) # result is a list of predictions, we will take the first element of the list
    

        return render_template('home.html',results=result[0])
    else:
        return render_template("home.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0")


# app.py is the main Python file that creates and runs your web application.

# In your screenshot, it uses Flask:

# from flask import Flask

# This creates a Flask application:

# app = Flask(__name__)

# This part defines what happens when someone opens the website:

# @app.route("/")
# def hello_world():
#     return "<h1>Hello, World!</h1>"

# So, when you open the application in a browser, it displays:

# Hello, World!

# This starts the application:

# if __name__ == "__main__":
#     app.run(host="0.0.0.0")

# You can run it from the terminal:

# python app.py

# Then open the displayed local address, usually:

# http://127.0.0.1:5000

# Flask uses these defaults:

# host="0.0.0.0" makes the app accessible on all network interfaces.
# Flask’s default port is 5000.
# 127.0.0.1 means your own computer.

# Therefore, the local address is:

# http://127.0.0.1:5000

# You can also use:

# http://localhost:5000


#1. PS C:\Users\ashutosh.patil\OneDrive - UKG\Documents\Data Science\Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main\5-Step By Step Project Implementation With LifeCycle Of ML Projects\Workspace> python application.py
#2. 