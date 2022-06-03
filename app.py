from flask import Flask, jsonify, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# loading the saved model
sf = pickle.load(open('sf.sav', 'rb'))
ccp = pickle.load(open('ccp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/churn.html', methods=['POST','GET'])
def churn():
    if request.method=='POST':
        Gender = 0
        if request.form["Gender"] == 1:
            Gender = 1
        SeniorCitizen = 0
        if 'SeniorCitizen' in request.form:
            SeniorCitizen = 1
        Partner = 0
        if 'Partner' in request.form:
            Partner = 1
        Dependents = 0
        if 'Dependents' in request.form:
            Dependents = 1
        PaperlessBilling = 0
        if 'PaperlessBilling' in request.form:
            PaperlessBilling = 1
            
        MonthlyCharges = int(request.form["MonthlyCharges"])
        Tenure = int(request.form["Tenure"])
        TotalCharges = MonthlyCharges * Tenure
        
        PhoneService = 0
        if 'PhoneService' in request.form:
            PhoneService = 1
    
        MultipleLines = 0
        if 'MultipleLines' in request.form and PhoneService == 1:
            MultipleLines = 1
    
        InternetService_Fiberoptic = 0
        InternetService_No = 0
        if request.form["InternetService"] == 0:
            InternetService_No = 1
        elif request.form["InternetService"] == 2:
            InternetService_Fiberoptic = 1
    
        OnlineSecurity = 0
        if 'OnlineSecurity' in request.form and InternetService_No == 0:
            OnlineSecurity = 1
    
        OnlineBackup = 0
        if 'OnlineBackup' in request.form and InternetService_No == 0:
            OnlineBackup = 1
    
        DeviceProtection = 0
        if 'DeviceProtection' in request.form and InternetService_No == 0:
            DeviceProtection = 1
    
        TechSupport = 0
        if 'TechSupport' in request.form and InternetService_No == 0:
            TechSupport = 1
    
        StreamingTV = 0
        if 'StreamingTV' in request.form and InternetService_No == 0:
            StreamingTV = 1
    
        StreamingMovies = 0
        if 'StreamingMovies' in request.form and InternetService_No == 0:
            StreamingMovies = 1
    
        features = np.asarray([[Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
           DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
           InternetService_Fiberoptic, InternetService_No]])
    
        columns = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
           'InternetService_Fiber optic', 'InternetService_No']
    
        features = features.reshape(1, -1)
        prediction = ccp.predict(features)
        return render_template('churn_result.html', prediction = prediction)
    else:
        return render_template('churn.html')

@app.route('/sales.html', methods=['POST','GET'])
def sales():
    if request.method=='POST':
        Item_Weight = float(request.form['Item Weight'])
        Item_Fat_Content = float(request.form['Item Fat Content'])
        Item_Visibility = float(request.form['Item Visibility'])
        Item_Type = float(request.form['Item Type'])
        Item_MRP = float(request.form['Item MRP'])
        Outlet_Identifier = float(request.form['Outlet Identifier'])
        Outlet_Establishment_Year = float(request.form['Outlet Establishment Year'])
        Outlet_Size = float(request.form['Outlet Size'])
        Outlet_Location_Type = float(request.form['Outlet Location Type'])
        Outlet_Type = float(request.form['Outlet Type'])
        X = np.asarray([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Identifier, 
                      Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type]])
        X = X.reshape(1, -1)
        prediction = sf.predict(X)
        return render_template('sales_result.html', prediction = prediction)
    else:
        return render_template('sales.html')

if __name__ == "__main__":
    app.run(debug=True)