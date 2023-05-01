# -*- coding: utf-8 -*-
"""

"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import uvicorn
import os
app = FastAPI()
@app.get("/")
def root():
    return {"message": "Welcome to the medical prediction API!"}

class model_input(BaseModel):
    
    pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int       
        
# loading the saved model
diabetes_model = pickle.load(open(os.path.join('MULTI_DIEASES_PREDICTION', 'savedmodels', 'diabetes_model.sav'), 'rb'))

@app.post('/diabetes_prediction')
def diabetes_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    prediction = diabetes_model.predict([input_list])
    
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    









class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

heart_disease_model = pickle.load(open(os.path.join('MULTI_DIEASES_PREDICTION', 'savedmodels', 'heart_disease_model.sav'), 'rb'))
@app.post('/predict/heart_disease')
def predict_heart_disease(input_parameters : HeartDiseaseInput):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    age = input_dictionary['age']
    sex = input_dictionary['sex']
    cp = input_dictionary['cp']
    trestbps = input_dictionary['trestbps']
    chol = input_dictionary['chol']
    fbs = input_dictionary['fbs']
    restecg = input_dictionary['restecg']
    thalach = input_dictionary['thalach']
    exang = input_dictionary['exang']
    oldpeak = input_dictionary['oldpeak']
    slope = input_dictionary['slope']
    ca = input_dictionary['ca']
    thal = input_dictionary['thal']
     
     
    input_list = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang,oldpeak,slope,ca,thal]
    prediction = heart_disease_model.predict([input_list])
    
    if (prediction[0] == 0):
        return 'The person is not heartic'
    else:
        return 'The person is heartic'
    






    
class ParkinsonsInput(BaseModel):
    fo: float
    fhi: float
    flo: float
    Jitter_percent: float
    Jitter_Abs: float
    RAP: float
    PPQ: float
    DDP: float
    Shimmer: float
    Shimmer_dB: float
    APQ3: float
    APQ5: float
    APQ: float
    DDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    spread1: float
    spread2: float
    D2: float
    PPE: float
parkinsons_model = pickle.load(open(os.path.join('MULTI_DIEASES_PREDICTION', 'savedmodels', 'parkinsons_model.sav'), 'rb'))

@app.post('/predict/parkinsons')
def predict_heart_disease(input_parameters : ParkinsonsInput):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    fo = input_dictionary['fo']
    fhi = input_dictionary['fhi']
    flo = input_dictionary['flo']
    Jitter_percent = input_dictionary['Jitter_percent']
    Jitter_Abs = input_dictionary['Jitter_Abs']
    RAP = input_dictionary['RAP']
    PPQ = input_dictionary['PPQ']
    DDP = input_dictionary['DDP']
    Shimmer = input_dictionary['Shimmer']
    Shimmer_dB = input_dictionary['Shimmer_dB']
    APQ3 = input_dictionary['APQ3']
    APQ5 = input_dictionary['APQ5']
    APQ = input_dictionary['APQ']
    DDA = input_dictionary['DDA']
    NHR = input_dictionary['NHR']
    RPDE = input_dictionary['RPDE'] 
    DFA = input_dictionary['DFA']
    spread1 = input_dictionary['spread1'] 
    spread2 = input_dictionary['spread2'] 
    D2 = input_dictionary['D2']
    PPE = input_dictionary['PPE'] 
    input_list = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,RPDE,DFA,spread1,spread2,D2,PPE]
    prediction = parkinsons_model.predict([input_list])
    
    if (prediction[0] == 0):
        return 'The person is not parkinsons'
    else:
        return 'The person is parkinsons'  
    




class LungDiseaseInput(BaseModel):
    GENDER: int
    AGE: int
    SMOKE: int
    YF: int
    ANXIETY: int
    PP: int
    CD: int
    FAT: int
    ALLERGY: int
    WHEEZING: int
    ALCH: int
    COUGH: int
    SOB: int
    SD: int
    CP: int

lung_model = pickle.load(open(os.path.join('MULTI_DIEASES_PREDICTION', 'savedmodels', 'lung_PREDICTION.sav'), 'rb'))
   

@app.post('/predict/lung_disease')
def predict_heart_disease(input_parameters : LungDiseaseInput):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)


    AGE = input_dictionary['GENDER']
    SMOKE = input_dictionary['AGE']
    YF = input_dictionary['SMOKE']
    ANXIETY = input_dictionary['YF']                
    PP = input_dictionary['ANXIETY']
    CD = input_dictionary['PP']
    FAT = input_dictionary['CD']
    ALLERGY = input_dictionary['Shimmer']
    WHEEZING = input_dictionary['Shimmer_dB']
    ALCH = input_dictionary['APQ3']
    COUGH = input_dictionary['APQ5']
    SOB = input_dictionary['APQ']
    SD = input_dictionary['DDA']
    CP = input_dictionary['NHR']
    
   
    input_list = [AGE, SMOKE, YF, ANXIETY, PP, CD, FAT, ALLERGY,WHEEZING,ALCH,COUGH,SOB,SD,CP]
    prediction = lung_model.predict([input_list])

    if (prediction[0] == 0):
        return 'The person is not lung_disease'
    else:
        return 'The person is lung_disease' 








