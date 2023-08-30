import joblib
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from keras import models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Plotting the confusion matrix

def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, xticklabels=['0','1'], yticklabels=['0','1'], ax=ax)
    ax.set(ylabel="Actual y", xlabel="Predicted y");
    return fig

# Evaluate the model
def Evaluate (y_test, y_pred):

    metrics_dict = {
        'Accuracy': accuracy_score(y_test, y_pred),   # Calculates the accuracy score using y_test (true values) and y_pred (predicted values) and adds it to the dictionary
        'Precision': precision_score(y_test, y_pred), # Calculates the precision score using y_test and y_pred and adds it to the dictionary
        'Recall': recall_score(y_test, y_pred),       # Calculates the recall score using y_test and y_pred and adds it to the dictionary
        'F1': f1_score(y_test, y_pred),               # Calculates the F1 score using y_test and y_pred and adds it to the dictionary
        'ROC AUC': roc_auc_score(y_test, y_pred)      # Calculates the ROC AUC score using y_test and y_pred and adds it to the dictionary
    }
    # Returns the dictionary containing all the computed metrics
    return metrics_dict

#load our test datasets
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")


# Load the model from the pickel file
filename = 'models/spaceship_ML_model.pkl'
Ml_model = joblib.load(filename)

st.header("Preprocessors:")
image = Image.open('IMG/preprocessor.png')
st.image(image, caption='Diagram of preprocessors by colomn type')

st.header("Model:")
box = st.selectbox("Choose the Model:", ['CatBoost','Deep Learning Classifier'])
if box == 'CatBoost':
    y_pred = Ml_model.predict(X_test)

if box == 'Deep Learning Classifier':
    modelFileName = 'models/spaceship_DL_classifier.h5'
    DL_model = models.load_model(modelFileName)

    N_X_valid = Ml_model['preprocessor'].transform(X_test)
    y_pred = DL_model.predict(N_X_valid)
    y_pred = [True if value >= 0.5 else False for value in y_pred]

st.subheader(f"Model Evaluation for {box}:")
col1, col2 = st.columns(2, gap="medium")
with col1:
   st.write(f"Confusion Matrix:")
   fig = plot_confusion(y_test, y_pred)
   st.pyplot(fig)

with col2:
   st.write(f"Evaluation Metrics:")
   st.write(Evaluate(y_test,y_pred))
