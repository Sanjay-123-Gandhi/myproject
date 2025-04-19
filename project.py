import streamlit as st
import pandas as pd
import numpy as np
import re
import ollama
import os
import spacy
from gtts import gTTS
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(data):
    # Removing duplicates
    data = data.drop_duplicates()
    
    # Handling missing values: fill with mean for numeric columns, mode for categorical columns
    for column in data.columns:
        if data[column].isnull().any():
            if data[column].dtype == 'object':
                # Fill missing values for categorical columns with mode
                data[column].fillna(data[column].mode()[0], inplace=True)
            else:
                # Fill missing values for numeric columns with mean
                data[column].fillna(data[column].mean(), inplace=True)

    # Optionally, drop irrelevant columns if needed
    # data = data.drop(columns=['column_name_to_drop'], errors='ignore')
    
    return data

def Regressor():
    
    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    # Splitting features and target
    X = df.drop([y_col], axis=1)
    y = df[y_col]
    features = X.columns
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Function to evaluate models
    def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return r2, rmse
    
    # Train models only when button is pressed
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False

    if st.button("Train Models"):
        st.write("### Training Models...")
        progress_bar = st.progress(0)
        
        with st.status("Training models...") as status:
            st.session_state.model_lr = LinearRegression().fit(X_train, y_train)
            progress_bar.progress(25)
        
            st.session_state.model_rf = RandomForestRegressor().fit(X_train, y_train)
            progress_bar.progress(50)
        
            st.session_state.model_svr = SVR(kernel='rbf').fit(X_train, y_train)
            progress_bar.progress(75)
        
            st.session_state.model_xgb = XGBRegressor().fit(X_train, y_train)
            progress_bar.progress(100)
        
            status.update(label="Training completed!", state="complete")
            st.session_state.models_trained = True

    # Ensure models are trained before making predictions
    if st.session_state.models_trained:
        # Evaluate models
        r2_lr, rmse_lr = evaluate_model(st.session_state.model_lr, X_test, y_test)
        r2_rf, rmse_rf = evaluate_model(st.session_state.model_rf, X_test, y_test)
        r2_svr, rmse_svr = evaluate_model(st.session_state.model_svr, X_test, y_test)
        r2_xgb, rmse_xgb = evaluate_model(st.session_state.model_xgb, X_test, y_test)
    
        st.write("### Model Performance on Test Set")
        st.write(f"Linear Regression: R² = {r2_lr:.2f}, RMSE = {rmse_lr:.2f}")
        st.write(f"Random Forest: R² = {r2_rf:.2f}, RMSE = {rmse_rf:.2f}")
        st.write(f"SVR: R² = {r2_svr:.2f}, RMSE = {rmse_svr:.2f}")
        st.write(f"XGBoost: R² = {r2_xgb:.2f}, RMSE = {rmse_xgb:.2f}")
    
        # User input for predictions
        st.write("### Enter Feature Values")
        feature_values = {}
    
        for col in X.columns:
            if col in categorical_columns:
                feature_values[col] = st.selectbox(f"{col}", label_encoders[col].classes_)
            else:
                feature_values[col] = st.number_input(f"{col}", value=0.0)
    
        # Convert categorical inputs to numerical
        for col in categorical_columns:
            if feature_values[col] in label_encoders[col].classes_:
                feature_values[col] = label_encoders[col].transform([feature_values[col]])[0]
            else:
                feature_values[col] = 0  # Default value for unknown categories
    
        input_array = np.array([feature_values[col] for col in X.columns]).reshape(1, -1)
    
        # Make predictions
        if st.button(f"Predict {y_col}"):
            with st.status("Generating predictions...") as status:
                prediction_lr = st.session_state.model_lr.predict(input_array)[0]
                prediction_rf = st.session_state.model_rf.predict(input_array)[0]
                prediction_svr = st.session_state.model_svr.predict(input_array)[0]
                prediction_xgb = st.session_state.model_xgb.predict(input_array)[0]
                
                status.update(label="Predictions ready!", state="complete")
                
                # Display predictions
                st.write("### Predictions")
                st.write(f"Linear Regression Prediction: {prediction_lr}")
                st.write(f"Random Forest Prediction: {prediction_rf}")
                st.write(f"Support Vector Regression (SVR) Prediction: {prediction_svr}")
                st.write(f"XGBoost Prediction: {prediction_xgb}")

            prediction_lr = st.session_state.model_lr.predict(input_array)[0]
            prediction_rf = st.session_state.model_rf.predict(input_array)[0]
            prediction_svr = st.session_state.model_svr.predict(input_array)[0]
            prediction_xgb = st.session_state.model_xgb.predict(input_array)[0]

            
        prediction_lr = st.session_state.model_lr.predict(input_array)[0]
        prediction_rf = st.session_state.model_rf.predict(input_array)[0]
        prediction_svr = st.session_state.model_svr.predict(input_array)[0]
        prediction_xgb = st.session_state.model_xgb.predict(input_array)[0]

        max_r_squared=max(r2_lr,r2_rf,r2_svr,r2_xgb)
        if max_r_squared==r2_lr:
            prediction = st.session_state.model_lr.predict(input_array)[0]
            best_model =st.session_state.model_lr
            st.write("linear")

        elif  max_r_squared==r2_rf:
            prediction = st.session_state.model_rf.predict(input_array)[0]
            best_model =st.session_state.model_rf
            st.write("Random_")
        elif  max_r_squared==r2_xgb:
            prediction = st.session_state.model_xgb.predict(input_array)[0]
            best_model =st.session_state.model_xgb
            st.write("XGB")
        else:
            prediction = st.session_state.model_lr.predict(input_array)[0]
            best_model =st.session_state.model_svr
            st.write("SVR")

            
    def Suggestion():
        # User input for optimization
        apply_suggestions = st.radio("Would you like to apply the suggestions?", ("Yes", "No"))
        if apply_suggestions == "Yes":
            st.subheader("Optimization Settings")
            categorical_optimize = st.multiselect("Select categorical columns to optimize:", categorical_columns)
            continuous_optimize = st.multiselect("Select continuous features to adjust:", X.columns)
            budget_flexibility = st.slider("Select budget flexibility (%)", 0, 30, 1) / 100
            # Suggest improvements button
            if st.button("Suggest Improvements"):
                st.subheader("Suggestions for Optimization")
                if prediction is not None:
                    for col in categorical_optimize:
                        best_value = feature_values[col]
                        best_prediction = prediction
                        for unique_value in df[col].unique():
                            temp_features = feature_values.copy()
                            temp_features[col] = unique_value
                            temp_array = np.array([temp_features[c] for c in X.columns]).reshape(1, -1)
                            temp_prediction = best_model.predict(temp_array)[0]
                            if temp_prediction > best_prediction:
                                best_prediction = temp_prediction
                                best_value = unique_value
                        st.write(f"For '{col}', consider using: {label_encoders[col].inverse_transform([best_value])[0]}"
                                 f"({y_col}: {best_prediction:.2f}, actual value: {df.loc[df[col] == best_value, y_col].values[0]})")

                        
                
                    for col in continuous_optimize:
                        lower_value = feature_values[col] * (1 - budget_flexibility)
                        upper_value = feature_values[col] * (1 + budget_flexibility)
                        temp_features_lower = feature_values.copy()
                        temp_features_upper = feature_values.copy()
                        temp_features_lower[col] = lower_value
                        temp_features_upper[col] = upper_value
                        temp_array_lower = np.array([temp_features_lower[c] for c in X.columns]).reshape(1, -1)
                        temp_array_upper = np.array([temp_features_upper[c] for c in X.columns]).reshape(1, -1)
                        lower_prediction = best_model.predict(temp_array_lower)[0]
                        upper_prediction = best_model.predict(temp_array_upper)[0]
                        st.write(f"For '{col}', consider changing value between [{lower_value:.2f}, {upper_value:.2f}]")
                        st.write(f"{y_col} at {lower_value:.2f}: {lower_prediction:.2f}, at {upper_value:.2f}: {upper_prediction:.2f},"
                                 f"actual value: {df.loc[(df[col] >= lower_value) & (df[col] <= upper_value), y_col].mean():.2f}")
        else:
            st.write(f"No suggestions were applied for {y_col}.")
     
                    # Chatbot function
    
    # Chatbot function
    def chatbot():
        st.write("### AI Chatbot")
        # Add a dropdown to select the language
        language = st.selectbox("Select language for chatbot response:", ["English", "Tamil"])

        # Text input for user query
        user_input = st.text_input("Ask a question about model performance:")

        if user_input:
            lemmatized_words = {token.lemma_ for token in nlp(user_input.lower())}

            if any(word in lemmatized_words for word in ["maximum", "highest", "best", "top", "superior"]):
                prompt = f"""
            You are an AI assistant that analyzes machine learning models.

            Here are the R² scores of different models:
            - Linear Regression: {r2_lr}
            - Random Forest: {r2_rf}
            - SVR: {r2_svr}
            - XGBoost: {r2_xgb}
            Here are the predictions for {y_col}:
            - Linear Regression prediction: {prediction_lr:.2f}
            - Random Forest prediction: {prediction_rf:.2f}
            - SVR prediction: {prediction_svr:.2f}
            - XGBoost prediction: {prediction_xgb:.2f}

            Identify the model with the **highest** R² and provide the answer in this format:

            "Best model is <model name> with an R² of <highest value>. Your prediction for {y_col} is <model name prediction>"
            """
            elif any(word in lemmatized_words for word in ["minimum", "lowest", "worst", "poor", "bad", "inferior"]):
                prompt = f"""
            You are an AI assistant that analyzes machine learning models.

            Here are the R² scores of different models:
            - Linear Regression: {r2_lr}
            - Random Forest: {r2_rf}
            - SVR: {r2_svr}
            - XGBoost: {r2_xgb}

            Identify the model with the **lowest** R² and provide the answer in this format:

            "Worst model is <model name> with an R² of <lowest value>."
            """
            else:
                prompt = "I'm sorry, I couldn't determine what you're asking. Please specify if you want to know the best or worst model."

            response = ollama.chat(model='mistral', messages=[
                {"role": "system", "content": "You are an AI chatbot that follows instructions carefully and answers only based on the given prompt."},
                {"role": "user", "content": prompt}
            ])

            chatbot_response = response['message']['content']
            st.write(f"Chatbot: {chatbot_response}")

            # Use the language selection to determine the speech output language
            if language == "English":
                tts = gTTS(text=chatbot_response, lang='en')
            elif language == "Tamil":
                # Convert text to speech using gTTS
                tts = gTTS(text=chatbot_response, lang='ta')
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")
    chatbot()
    Suggestion()

       
    
    

def Classifier():

    #Encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_columns = categorical_columns.difference([y_col])
    label_encoders = {}

    # Encode the target column (y)
    label_encoder_y = LabelEncoder()
    df[y_col] = label_encoder_y.fit_transform(df[y_col])

        
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    #Splitting features and target
    X = df.drop([y_col], axis=1)
    y = df[y_col]
    features = X.columns

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Function to evaluate models
    def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        return accuracy, f1

    # Train models only when button is pressed
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
        
    if st.button("Train Models"):
        st.write("### Training Models...")
        progress_bar = st.progress(0)
            
        with st.status("Training models...") as status:
            st.session_state.model_lr = LogisticRegression().fit(X_train, y_train)
            progress_bar.progress(25)
                
            st.session_state.model_rf = RandomForestClassifier().fit(X_train, y_train)
            progress_bar.progress(50)
                
            st.session_state.model_svc = SVC(kernel='rbf').fit(X_train, y_train)
            progress_bar.progress(75)
                
            st.session_state.model_xgb = XGBClassifier().fit(X_train, y_train)
            progress_bar.progress(100)
                
            status.update(label="Training completed!", state="complete")
            st.session_state.models_trained = True

    # Ensure models are trained before making predictions
    if st.session_state.models_trained:
        # Evaluate models
        accuracy_lr, f1_lr = evaluate_model(st.session_state.model_lr, X_test, y_test)
        accuracy_rf, f1_rf = evaluate_model(st.session_state.model_rf, X_test, y_test)
        accuracy_svc, f1_svc = evaluate_model(st.session_state.model_svc, X_test, y_test)
        accuracy_xgb, f1_xgb = evaluate_model(st.session_state.model_xgb, X_test, y_test)
            
        st.write("### Model Performance on Test Set")
        st.write(f"Logistic Regression: Accuracy = {accuracy_lr:.2f}, F1 Score = {f1_lr:.2f}")
        st.write(f"Random Forest Classifier: Accuracy = {accuracy_rf:.2f}, F1 Score = {f1_rf:.2f}")
        st.write(f"Support Vector Classifier (SVC): Accuracy = {accuracy_svc:.2f}, F1 Score = {f1_svc:.2f}")
        st.write(f"XGBoost Classifier: Accuracy = {accuracy_xgb:.2f}, F1 Score = {f1_xgb:.2f}")

        # User input for predictions
        st.write("### Enter Feature Values")
        feature_values = {}
            
        for col in X.columns:
            if col in categorical_columns:
                feature_values[col] = st.selectbox(f"{col}", label_encoders[col].classes_)
            else:
                feature_values[col] = st.number_input(f"{col}", value=0.0)
            
        # Convert categorical inputs to numerical
        for col in categorical_columns:
            if feature_values[col] in label_encoders[col].classes_:
                feature_values[col] = label_encoders[col].transform([feature_values[col]])[0]
            else:
                feature_values[col] = 0  # Default value for unknown categories
            
        input_array = np.array([feature_values[col] for col in X.columns]).reshape(1, -1)

        # Make predictions
        if st.button(f"Predict {y_col}"):
            with st.status("Generating predictions...") as status:
                prediction_lr = st.session_state.model_lr.predict(input_array)[0]
                prediction_rf = st.session_state.model_rf.predict(input_array)[0]
                prediction_svc = st.session_state.model_svc.predict(input_array)[0]
                prediction_xgb = st.session_state.model_xgb.predict(input_array)[0]
                    
                prediction_lr = label_encoder_y.inverse_transform([prediction_lr])[0]
                prediction_rf = label_encoder_y.inverse_transform([prediction_rf])[0]
                prediction_svc = label_encoder_y.inverse_transform([prediction_svc])[0]
                prediction_xgb = label_encoder_y.inverse_transform([prediction_xgb])[0]


                status.update(label="Predictions ready!", state="complete")

                # Show predictions
                st.write("### Predictions")
                st.write(f"Logistic Regression Prediction: {prediction_lr}")
                st.write(f"Random Forest Prediction: {prediction_rf}")
                st.write(f"Support Vector Classifier Prediction: {prediction_svc}")
                st.write(f"XGBoost Prediction: {prediction_xgb}")

            prediction_lr = st.session_state.model_lr.predict(input_array)[0]
            prediction_rf = st.session_state.model_rf.predict(input_array)[0]
            prediction_svc = st.session_state.model_svc.predict(input_array)[0]
            prediction_xgb = st.session_state.model_xgb.predict(input_array)[0]

        prediction_lr = st.session_state.model_lr.predict(input_array)[0]
        prediction_rf = st.session_state.model_rf.predict(input_array)[0]
        prediction_svc = st.session_state.model_svc.predict(input_array)[0]
        prediction_xgb = st.session_state.model_xgb.predict(input_array)[0]


        max_accuracy_score=max(accuracy_lr,accuracy_rf,accuracy_svc,accuracy_xgb)
        if max_accuracy_score==accuracy_lr:
            prediction = st.session_state.model_lr.predict(input_array)[0]
            best_model =st.session_state.model_lr
            st.write("Logestic Regression")

        elif  max_accuracy_score==accuracy_rf:
            prediction = st.session_state.model_rf.predict(input_array)[0]
            best_model =st.session_state.model_rf
            st.write("Random Forest Classifier ")
        elif  max_accuracy_score==accuracy_xgb:
            prediction = st.session_state.model_xgb.predict(input_array)[0]
            best_model =st.session_state.model_xgb
            st.write("XGB Classifier")
        else:
            prediction = st.session_state.model_lr.predict(input_array)[0]
            best_model =st.session_state.model_svc
            st.write("SVC")
    def Suggestion():
        # User input for optimization
        apply_suggestions = st.radio("Would you like to apply the suggestions?", ("Yes", "No"))
        if apply_suggestions == "Yes":
            st.subheader("Optimization Settings")
            categorical_optimize = st.multiselect("Select categorical columns to optimize:", categorical_columns)
            continuous_optimize = st.multiselect("Select continuous features to adjust:", X.columns)
            budget_flexibility = st.slider("Select budget flexibility (%)", 0, 30, 1) / 100
            # Suggest improvements button
            if st.button("Suggest Improvements"):
                st.subheader("Suggestions for Optimization")
                if prediction is not None:
                    for col in categorical_optimize:
                        best_value = feature_values[col]
                        best_prediction = prediction
                        for unique_value in df[col].unique():
                            temp_features = feature_values.copy()
                            temp_features[col] = unique_value
                            temp_array = np.array([temp_features[c] for c in X.columns]).reshape(1, -1)
                            temp_prediction = best_model.predict(temp_array)[0]
                            if temp_prediction > best_prediction:
                               best_prediction = temp_prediction
                               best_value = unique_value
                        st.write(f"For '{col}', consider using: {label_encoders[col].inverse_transform([best_value])[0]}"
                                f"({y_col}: {best_prediction:.2f}, actual value: {df.loc[df[col] == best_value, y_col].values[0]})")

   
                    for col in continuous_optimize:
                        lower_value = feature_values[col] * (1 - budget_flexibility)
                        upper_value = feature_values[col] * (1 + budget_flexibility)
                        temp_features_lower = feature_values.copy()
                        temp_features_upper = feature_values.copy()
                        temp_features_lower[col] = lower_value
                        temp_features_upper[col] = upper_value
                        temp_array_lower = np.array([temp_features_lower[c] for c in X.columns]).reshape(1, -1)
                        temp_array_upper = np.array([temp_features_upper[c] for c in X.columns]).reshape(1, -1)
                        lower_prediction = best_model.predict(temp_array_lower)[0]
                        upper_prediction = best_model.predict(temp_array_upper)[0]
                        st.write(f"For '{col}', consider changing value between [{lower_value:.2f}, {upper_value:.2f}]")
                        st.write(f"{y_col} at {lower_value:.2f}: {lower_prediction:.2f}, at {upper_value:.2f}: {upper_prediction:.2f},"
                                 f"actual value: {df.loc[(df[col] >= lower_value) & (df[col] <= upper_value), y_col].mean():.2f}")
        else:
            st.write(f"No suggestions were applied for {y_col}.")
    Suggestion()
                    
    #chatbot funtion
    def chatbot():
        st.write("### AI Chatbot")

        # Add a dropdown to select the language
        language = st.selectbox("Select language for chatbot response:", ["English", "Tamil"])

        # Text input for user query
        user_input = st.text_input("Ask a question about model performance:")
        if user_input:
            lemmatized_words = {token.lemma_ for token in nlp(user_input.lower())}

            if any(word in lemmatized_words for word in ["maximum", "highest", "best", "top", "superior"]):
                prompt = f"""
            You are an AI assistant that analyzes machine learning models.

            Here are the Accuracy of different models:
            - Logistic Regression: {accuracy_lr}
            - Random Forest: {accuracy_rf}
            - SVC: {accuracy_svc}
            - XGBoost: {accuracy_xgb}
            Here are the predictions for {y_col}:
            - Logistic Regression prediction: {label_encoder_y.inverse_transform([prediction_lr])[0]}
            - Random Forest prediction: {label_encoder_y.inverse_transform([prediction_rf])[0]}
            - SVC prediction: {label_encoder_y.inverse_transform([prediction_svc])[0]}
            - XGBoost prediction: {label_encoder_y.inverse_transform([prediction_xgb])[0]}


            Identify the model with the **highest** Accuracy and provide the answer in this format:

            "Best model is <model name> with an Accuracy of <highest accuracy>.Your prediction for {y_col} is <model name prediction>"
            """
            elif any(word in lemmatized_words for word in ["minimum", "lowest", "worst", "poor", "bad", "inferior"]):
                prompt = f"""
            You are an AI assistant that analyzes machine learning models.

            Here are the Accuracy scores of different models:
            - Logistic Regression: {accuracy_lr}
            - Random Forest: {accuracy_rf}
            - SVC: {accuracy_svc}
            - XGBoost: {accuracy_xgb}

            Identify the model with the **lowest** Accuracy and provide the answer in this format:

            "Worst model is <model name> with an Accuracy of <lowest accuracy>."
            """
            else:
                prompt = "I'm sorry, I couldn't determine what you're asking. Please specify if you want to know the best or worst model."

            response = ollama.chat(model='mistral', messages=[
                {"role": "system", "content": "You are an AI chatbot that follows instructions carefully and answers only based on the given prompt."},
                {"role": "user", "content": prompt}
            ])

            chatbot_response = response['message']['content']
            st.write(f"Chatbot: {chatbot_response}")

            # Use the language selection to determine the speech output language
            if language == "English":
                tts = gTTS(text=chatbot_response, lang='en')
            elif language == "Tamil":
                # Convert text to speech using gTTS
                tts = gTTS(text=chatbot_response, lang='ta')

            # Save and play the audio response
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")
    chatbot()
    

    
        
def plotting():

    st.title("DYNAMIC GRAPH PLOTTER")

    # Select X and Y axes
    x = st.selectbox("Select your x-axis", df.columns)
    y = st.selectbox("Select your y-axis", df.columns)

    # Button to trigger the graph plot
    if st.button("Plot Graph"):
        # Check if 'x' contains 'year' or 'years' using regex
        pattern = r'\b(year|years)\b|year|years'
        matches = re.search(pattern, x.lower())# Use search instead of findall (faster for a single match)

        plt.figure(figsize=(10, 6))

        if matches:  # Line graph for year-based data
            plt.plot(df[x], df[y], marker='o', markersize=8, markerfacecolor='yellow', linestyle='-', color='red')
            plt.title(f"Line Graph of {y} over {x}")

        elif df[x].dtype == 'object':  # Categorical feature
            sns.barplot(x=df[x], y=df[y], estimator=sum, ci=None)
            plt.xticks(rotation=45)
            plt.title(f"Bar Chart of {y} by {x}")

        else:  # Continuous feature (Scatter plot)
            plt.scatter(df[x], df[y], marker='o', c='r')
            plt.title(f"Scatter Plot of {y} vs {x}")

        # Fixed xlabel and ylabel syntax
        plt.xlabel(x, color='green', fontsize=20)
        plt.ylabel(y, color='green', fontsize=20)

        st.pyplot(plt)  # Show the plot in Streamlit
    
# Initialize NLP and TTS engine
nlp = spacy.load("en_core_web_sm")

# Streamlit app title
st.title("REAL-TIME DYNAMIC DATA PREDICTIONS POWERED BY MACHINE LEARNING AND CHATBOT")
st.write("""Instructions for using our app:\n
Step 1: Choose the file type you want to upload.\n
Step 2: Choose the file you want to upload.\n
Step 3: Choose two fields for checking their relationship between.\n
Step 4: Choose the field you want to predict.\n
Step 5: Train the model.\n
Step 6: Fill in the values for fields to get predictions.\n
Step 7: Ask the chatbot for the model that fits best for your dataset or worst model for your dataset
you can also ask metrics of the model used.\n
Step 8: Choose the field and budget flexibility to get Suggestions\n""")

# Upload file
file_types = ["csv", "xlsx"]
st.write("Our app supports only Excel (.xlsx) and CSV files")

select_file_type = st.selectbox("Select the file type:", file_types)

uploaded_file = st.file_uploader("Upload your file", type=file_types)

if uploaded_file is not None:
    try:
        if select_file_type == "csv":
            df = pd.read_csv(uploaded_file)
        else:  # If file is Excel
            df = pd.read_excel(uploaded_file)

        # Clean the data
        df = clean_data(df)

        st.write("### Thank you for using our Prediction App")
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        plotting()
        # Ask user for the target column
        y_col = st.selectbox("Select the target column to classify:", df.columns)
        if df[y_col].dtype == 'object':
            Classifier()
        else:
            Regressor()
    except Exception as e:
        st.error(f"Error reading file: {e}") 
