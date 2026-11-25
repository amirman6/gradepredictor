# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:19:05 2022

@author: T430s
"""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from streamlit_option_menu import option_menu

st.write("""
#### UP-2 Final Exam Score Prediction and Analysis App
--by A. Maharjan 
""")
image = Image.open('rit.jpg') # if errors, make sure to convert any image to jpg eg using online help
st.image(image)


selected = option_menu(
    menu_title= None,
    options= ["Final Exam Score Prediction","Analytics"],
    menu_icon = "cast",
    #default_index = 0,
    orientation= "horizontal",
    styles = {
        "nav-link":{"font-size":"14px","text-align":"center"},
        "nav-link-selected": {"background-color":"green"}
        }
    
    )




   
    

if selected == "Final Exam Score Prediction":
    st.write('(Choose the variables from the  LEFT SIDE BAR)')
    # loading the saved model for Home Price(random forest model was used)
    #df_data = pickle.load(open('df_data','rb'))
    model_final = pickle.load(open('model_finalexam','rb'))
    

    
    #st.write('---')

    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')
   # st.sidebar.write('---')

    st.sidebar.header('Exam-1 Score')
    exam1 = st.sidebar.number_input('Input your Exam-1 score',step = 1)
    #st.sidebar.write('---')
    
    st.sidebar.header('Exam-2 Score')
    exam2 = st.sidebar.number_input('Input your Exam-2 score',step = 1)
    #st.sidebar.write('---')
    
    st.sidebar.header('Exam-3 Score')
    exam3 = st.sidebar.number_input('Input your Exam-3 score',step = 1)
    #st.sidebar.write('---')
    
    st.sidebar.header('Workshop + Lab')
    workshop_lab = st.sidebar.number_input('Input your Workshop+Labs Average score',step = 1)
    #st.sidebar.write('---')
    
    st.sidebar.header('ExpertTA HW')
    ExpertTA_HW = st.sidebar.number_input('Input your ExpertTA HW average score',step = 1)
    #st.sidebar.write('---')


    # dataframe for model prediction
    X_new = pd.DataFrame([[exam1,exam2,exam3,workshop_lab,ExpertTA_HW]],
                         columns=['Exam1', 'Exam2', 'Exam3', 'Workshop_Labs_avg', 'ExpertTA_HW_avg'])



    
    
    # for monte-carlo simulations and prediction 
    if st.button('Predict'):
        sample = X_new.iloc[0]
        n_simulations = 5000
        test_mean3 = sample['Exam3'] # exam3 is the most influncer/important feature
        test_std3 = test_mean3 * 0.1  # 10% standard deviation
        test_mean2 = sample['Exam2'] # exam2 is the 2nd most influncer/important feature
        test_std2 = test_mean2 * 0.1  # 10% standard deviation
        #test_mean1 = sample['Exam1'] # exam2 is the 3rd most influncer/important feature
        #test_std1 = test_mean1 * 0.1  # 10% standard deviation
        
        simulated_samples = pd.DataFrame([sample.copy() for _ in range(n_simulations)]) # randomly chooses 5000 points from the sample data
        simulated_samples['Exam3'] = np.random.normal(test_mean3, test_std3, n_simulations) # brings slight variation in Exam3 based on hte std choosen.
        simulated_samples['Exam2'] = np.random.normal(test_mean2, test_std2, n_simulations) # brings slight variation in Exam3 based on hte std choosen.
        #simulated_samples['Exam1'] = np.random.normal(test_mean1, test_std1, n_simulations)
        # returns normally distributed list of data
        
        predictions = model_final.predict(simulated_samples) # xgboost provides better among three so far.
        avg_score = np.round(np.mean(predictions),1)
        standard_deviation = np.round(np.std(predictions),1)
        print ('The average predicted final exam score is:', avg_score,"+/-", standard_deviation )

        # results output
        st.write('-Estimated Predicted Final Exam Score is:', avg_score, '+/-',standard_deviation)
        fig,ax = plt.subplots()
        plt.hist(predictions, bins=20, edgecolor='k', alpha=0.7)
        plt.title('Monte Carlo Simulation of Final Exam Score Prediction')
        plt.xlabel('Predicted Final Exam Score')
        plt.ylabel('Frequency')
        plt.axvline(np.median(predictions), color='red', linestyle='dashed', linewidth=2, label='Median Prediction')
        plt.axvline(np.mean(predictions), color='blue', linestyle='dashed', linewidth=2, label='Mean Prediction')
        plt.legend()
        #plt.show()
        st.pyplot(fig)
    


if selected == "Analytics":
    model_final = pickle.load(open('model_finalexam','rb'))
    df = pickle.load(open('df','rb'))
    if st.button('PRESS to See Which Category Affect the Most'):
        
        
        st.write("""
        ##### Important Features 
        """)
        image = Image.open('feature_importance.jpg') # if errors, make sure to convert any image to jpg eg using online help
        st.image(image)
        
       
    st.write('---------')
    st.write("""
    ##### Select below to see the relationship with Overall Grade 
    """)  
    
    X = st.selectbox((''),('','Workshop_Labs_avg', 'ExpertTA_HW_avg','Exam1', 'Exam2', 'Exam3')) 
    if st.button('PRESS to see the relationship'):
        df_categories = df[['Workshop_Labs_avg', 'ExpertTA_HW_avg','Exam1', 'Exam2', 'Exam3']]
        fig,ax = plt.subplots()
        sns.scatterplot(data = df_categories,x = X,y = df.Overall,hue = df.CourseGrade)
        plt.xlabel(X)
        plt.ylabel('Overall Grade')
        st.pyplot(fig)
    
   
         
         
    
    

    


    

     
       




