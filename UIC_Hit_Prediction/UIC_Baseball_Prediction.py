import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import os
import json


# Load the dataset into a pandas DataFrame
df = pd.read_csv('/Users/purva/Documents/IDS560_Capstone/Data/UIC_baseball.csv', low_memory=False)

# Drop all rows with no PitcherId or BatterId

df.dropna(subset=['PitcherId', 'BatterId'], how='any')

# Define columns to encode
columns_to_encode = ['PitcherThrows','BatterSide','PitcherSet','Top/Bottom','TaggedPitchType','AutoPitchType','PitchCall','KorBB','TaggedHitType','PlayResult']

# Create a LabelEncoder instance for each column to encode
label_encoders = {col: LabelEncoder() for col in columns_to_encode}

# Apply the LabelEncoder to each column in a new dataframe

df_encoding = df.copy()
for col, encoder in label_encoders.items():
    df_encoding[col] = encoder.fit_transform(df[col])

df_dropped = df.copy()
df_encoding_dropped = df_encoding.copy()

df_dropped_uic_pitcher = df_dropped.copy()
df_dropped_uic_batter = df_dropped.copy()
df_dropped_uic_pitcher = df_dropped_uic_pitcher[df_dropped_uic_pitcher['PitcherTeam'] == "UIC_FLA"]
df_dropped_uic_batter = df_dropped_uic_batter[df_dropped_uic_batter['BatterTeam'] == "UIC_FLA"]

df_dropped_uic = pd.concat([df_dropped_uic_batter, df_dropped_uic_pitcher], ignore_index=True)

# UIC Dataframe

df_encoding_dropped_uic_pitcher = df_encoding_dropped.copy()
df_encoding_dropped_uic_batter = df_encoding_dropped.copy()
df_encoding_dropped_uic_pitcher = df_encoding_dropped_uic_pitcher[df_encoding_dropped_uic_pitcher['PitcherTeam'] == "UIC_FLA"]
df_encoding_dropped_uic_batter = df_encoding_dropped_uic_batter[df_encoding_dropped_uic_batter['BatterTeam'] == "UIC_FLA"]

df_encoding_dropped_uic = pd.concat([df_encoding_dropped_uic_batter, df_encoding_dropped_uic_pitcher], ignore_index=True)


df_dropped_uic_clean = df_dropped_uic.copy() 
df_encoding_dropped_uic_clean = df_encoding_dropped_uic.copy() 

df_dropped_uic_clean['Hit'] = df_dropped_uic_clean['ExitSpeed'].notnull().astype(int)
df_encoding_dropped_uic_clean['Hit'] = df_encoding_dropped_uic_clean['ExitSpeed'].notnull().astype(int)

columns_to_keep = ['PlayResult','Top/Bottom', 'TaggedPitchType', 'PitchCall', 'KorBB', 'TaggedHitType', 'OutsOnPlay',  'RunsScored', 'RelSpeed', 'HorzRelAngle', 'SpinRate', 'PlateLocHeight', 'PlateLocSide', 'VertApprAngle', 'ExitSpeed', 'Angle', 'Direction', 'HitSpinRate', 'vz0', 'ay0', 'BatterSide', 'PitcherThrows', 'Extension', 'ZoneSpeed', 'RelHeight', 'MaxHeight', 'InducedVertBreak', 'LastTrackedDistance', 'Hit', 'Pitcher', 'PitcherId', 
'Batter', 'BatterId', 'PitcherTeam', 'BatterTeam'] 

all_columns = df_encoding_dropped_uic_clean.columns.tolist() 

columns_to_drop1 = [col for col in all_columns if col not in columns_to_keep] 

df_encoding_dropped_uic_clean = df_encoding_dropped_uic_clean.drop(columns=columns_to_drop1) 
df_dropped_uic_clean = df_dropped_uic_clean.drop(columns=columns_to_drop1) 


df_uic_model = df_encoding_dropped_uic_clean[df_encoding_dropped_uic_clean['BatterTeam'] == 'UIC_FLA']

unique_batters_uic = df_uic_model[['Batter', 'BatterId']].drop_duplicates()

# Create the mapping
name_to_id_uic = pd.Series(unique_batters_uic.BatterId.values, index=unique_batters_uic.Batter).to_dict()

# Extract unique batter names
batter_names_uic = unique_batters_uic['Batter'].unique().tolist()

# Send to JSON for mapping dictionary
with open('/Users/purva/Documents/IDS560_Capstone/Code/Models/batter_name_to_id_mapping.json', 'w') as f:
    json.dump(name_to_id_uic, f)

general_data = []

feature_cols = ['RelSpeed', 'SpinRate', 'PlateLocHeight', 'PlateLocSide'] 

# Directory to save models and scalers
models_dir = '/Users/purva/Documents/IDS560_Capstone/Code/Models' 
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Training and saving models
feature_cols = ['RelSpeed', 'SpinRate', 'PlateLocHeight', 'PlateLocSide']
for batter_name, batter_id in name_to_id_uic.items():
    batter_data = df_dropped_uic_clean[df_dropped_uic_clean['BatterId'] == batter_id]
    if len(batter_data) >= 10:  # Ensure sufficient data for training
        X = batter_data[feature_cols]
        y = batter_data['Hit']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler().fit(X_train)
        model = RandomForestClassifier(random_state=42).fit(scaler.transform(X_train), y_train)
        # Save the model and scaler
        dump(model, os.path.join(models_dir, f'{batter_id}_model.joblib'))
        dump(scaler, os.path.join(models_dir, f'{batter_id}_scaler.joblib'))

# Prepare data for training the general model
if general_data:
    general_df = pd.concat(general_data)
    X_general = general_df[feature_cols]
    y_general = general_df['Hit']

    # Splitting the general dataset
    X_train_general, X_test_general, y_train_general, y_test_general = train_test_split(X_general, y_general, test_size=0.3, random_state=42)

    # Scaling the general dataset features
    scaler_general = StandardScaler().fit(X_train_general)
    X_train_general_scaled = scaler_general.transform(X_train_general)
    X_test_general_scaled = scaler_general.transform(X_test_general)

    # Training the general model
    general_model = RandomForestClassifier(random_state=42)
    general_model.fit(X_train_general_scaled, y_train_general)

    # Save the general model and scaler to disk
    dump(general_model, os.path.join(models_dir, 'general_model.joblib'))
    dump(scaler_general, os.path.join(models_dir, 'general_scaler.joblib'))

    y_pred_general = general_model.predict(X_test_general_scaled)
    print(f'Accuracy for the general model: {accuracy_score(y_test_general, y_pred_general)}')
else:
    print("No general data available for training a general model.")
