from flask import Flask, request, app, render_template
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app = application

preprocessor = pickle.load(open('Model/preprocessor.pkl','rb'))
model = pickle.load(open("Model/model.pkl",'rb'))


@app.route("/")
def hello_world():
    return render_template('index.html')

# Route for single data point prediction
@app.route('/predictdata', methods=['GET','POST'])

def predict_datapoint():
    result = ''
    if request.method=='POST':

        Age = float(request.form.get("Age"))
        Gender = str(request.form.get('Gender'))
        Location = str(request.form.get('Location'))
        Subscription_Length_Months = float(request.form.get('Subscription_Length_Months'))
        Monthly_Bill = float(request.form.get('Monthly_Bill'))
        Total_Usage_GB = float(request.form.get('Total_Usage_GB'))
        
        new_data=preprocessor.transform([[Age, Gender, Location, Subscription_Length_Months, Monthly_Bill, Total_Usage_GB]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Churn = 1'
        else:
            result ='Churn = 0'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
