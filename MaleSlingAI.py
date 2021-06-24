#Core Pkgs
import streamlit as st

#EDA Pkgs
import pandas as pd
import numpy as np

#Utils
import os
import joblib
import hashlib
#passlib,bcrypt

#Data Viz Pckgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# DB
from managed_db import *

#Password
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False

feature_names_best = ['Diabetes', 'FormerSmoker0no1yes2current','Race1white2black3other', 'AgeatAdVance', 'Heightatslingsurgerycm',
       'weightatslingsurgerykg', 'Prostatectomy1yes0no', 'PelvicRads1yes0no','Prior_SUI_mgt', 'BNC', 'Incontinence_Type',
       'code_ConcomitantProcedures','Preop','Etiology1Prostatectomy2otherprostatetherapy3radiation4NGBSCI5TUR']

Diabetes_dict = {"History of Diabetes":1, "No Diabetes":2}
FormerSmoker0no1yes2current_dict = {"No smoking history":0, "Previous smoker":1, "Current smoker":2}
Race1white2black3other_dict = {"White":1,"Black":2, "Other":3}
Incontinence_Type_dict = {"Stress Urinary Incontinence":1, "Mixed Incontinence":2}
code_ConcomitantProcedures_dict = {"No concomitant procedures planned":0, "Concomitant procedures planned":1}
Etiology1Prostatectomy2otherprostatetherapy3radiation4NGBSCI5TUR_dict = {"Prostatectomy":1, "Other prostate therapy":2, "Radiation":3, "Neurogenic bladder/Spinal cord injury":4, "Transurethral resecctions":5}

def get_value(val,my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val):
    for key, value in my_dict.items():
        if val == key:
            return value
        
def get_feature_value(val):
    feature_dict = {"Yes":1, "No":2}
    for key, value in feature_dict.items():
        if val == key:
            return value

# Load ML models        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """Male Sling Success Prediction"""
    st.title("Male Sling Success Prediction")
    
    menu = ["Home"]
    submenu = ["Prediction"]
    
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text ("Prediction of success of AdVance Male Sling")
                
        activity = st.selectbox("Activity", submenu)
        if activity == "Prediction":
                    st.header("Predictive Analytics")
                    
                    st.subheader ("Patient characteristics")
                    
                    AgeatAdVance = st.number_input("Age (Years)",18,99)
                    Heightatslingsurgerycm = st.number_input("Height (cm)",160,200)
                    weightatslingsurgerykg = st.number_input("Weight (kg)",45,130)
                    Diabetes = st.radio("Diabetes",tuple(Diabetes_dict.keys()))
                    FormerSmoker0no1yes2current = st.radio("Smoking history",tuple(FormerSmoker0no1yes2current_dict.keys()))
                    Race1white2black3other = st.radio("Race",tuple(Race1white2black3other_dict.keys()))
                    BNC = st.radio("Bladder neck contracture",tuple(feature_dict.keys()))
                    
                    st.subheader ("Incontinence characteristics")
                    
                    Incontinence_Type = st.radio("Type of Incontiennce",tuple(Incontinence_Type_dict.keys()))
                    Etiology1Prostatectomy2otherprostatetherapy3radiation4NGBSCI5TUR = st.radio("Etiology of incontinence",tuple(Etiology1Prostatectomy2otherprostatetherapy3radiation4NGBSCI5TUR_dict.keys()))
                    Preop = st.number_input("Number of pads per day",1,12)
                
                    st.subheader ("Prior interventions")
                    Prostatectomy1yes0no = st.radio("Prostatectomy", tuple(feature_dict.keys()))
                    PelvicRads1yes0no = st.radio("Pelvic Radiation", tuple(feature_dict.keys()))
                    Prior_SUI_mgt = st.radio("Prior stress urinary incontinence management", tuple(feature_dict.keys()))
                    
                    feature_list = [AgeatAdVance, Heightatslingsurgerycm, weightatslingsurgerykg, get_value(Diabetes,Diabetes_dict), 
                                    get_value(FormerSmoker0no1yes2current,FormerSmoker0no1yes2current_dict), get_value(Race1white2black3other, Race1white2black3other_dict), 
                                    get_feature_value(BNC), get_value(Incontinence_Type,Incontinence_Type_dict), get_value(Etiology1Prostatectomy2otherprostatetherapy3radiation4NGBSCI5TUR,Etiology1Prostatectomy2otherprostatetherapy3radiation4NGBSCI5TUR_dict),
                                    Preop, get_feature_value(Prostatectomy1yes0no), get_feature_value(PelvicRads1yes0no), get_feature_value(Prior_SUI_mgt)]
                   
                    st.write(feature_list)
                    pretty_result = {"Age":AgeatAdVance, "Height":Heightatslingsurgerycm, "Weight":weightatslingsurgerykg, "Diabetes":Diabetes, "Smoking history":FormerSmoker0no1yes2current, "Race":Race1white2black3other,
                                     "Bladder neck contracture":BNC, "Type of Incontinence":Incontinence_Type, "Etiology of incontinence":Etiology1Prostatectomy2otherprostatetherapy3radiation4NGBSCI5TUR, 
                                     "Number of pads per day":Preop, "Prior prostatectomy":Prostatectomy1yes0no, "Prior radiation":PelvicRads1yes0no, "Prior stress urinary incontinence management": Prior_SUI_mgt}
                    st.json(pretty_result)
                    single_sample = np.array(feature_list).reshape(1,-1)
                 
                    #ML
                    model_choice = st.selectbox("Select Model", ["Decision Tree"])
                    if st.button("Predict"):
                        if model_choice == "Decision Tree":
                            loaded_model = load_model("Pickle_Dec_MaleSling.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                            
                        st.write(prediction)
                        if prediction == 1:
                            st.warning("The surgery will be successful")
                            pred_probability_score = {"Failure":pred_prob[0][1]*100, "Success":pred_prob[0][0]*100}
                            st.subheader("Prediction Probability Score using {}".format(model_choice))
                            st.json(pred_probability_score)
                        elif prediction == 0:
                            st.success("The surgery will not be successful")
                            pred_probability_score = {"Failure":pred_prob[0][1]*100, "Success":pred_prob[0][0]*100}
                            st.subheader("Prediction Probability Score using {}".format(model_choice))
                            st.json(pred_probability_score)
                            
    
                               


if __name__ == '__main__':
    main()
