import pandas as pd
import os
import zipfile

with zipfile.ZipFile("dataset_cleaned.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset_cleaned_extracted")
valid = pd.read_csv(os.path.join("dataset_cleaned_extracted","valid.csv"))

x_columns = ['Male', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
       'PhysicalActivities', 'SleepHours', 'RemovedTeeth',
       'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
       'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
       'ECigaretteUsage', 'ChestScan', 'HeightInMeters', 'WeightInKilograms',
       'BMI', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
       'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos']
y_column = 'HadHeartAttack'


valid_x = valid[x_columns]
valid_y = valid[y_column]

from tensorflow import keras
model = keras.models.load_model('model.keras')

import numpy as np
predictions = model.predict(valid_x)[:,0]
true_answers = valid_y.to_numpy()
validation_accuracy = np.sum(np.rint(predictions) == true_answers)/len(true_answers)
print(f"Poprawność na zbiorze walidacyjnym: {validation_accuracy:.2%}")

np.savetxt("predictions.txt",predictions)
np.savetxt("predictions_two_digits.txt",predictions, fmt='%1.2f')

validate_heart_disease_true = valid.loc[valid[y_column]==1]
validate_heart_disease_false = valid.loc[valid[y_column]==0]

from datetime import timezone
import datetime
import json

validate_heart_disease_true_x = validate_heart_disease_true[x_columns]
validate_heart_disease_false_x = validate_heart_disease_false[x_columns]



predictions_for_true = model.predict(validate_heart_disease_true_x)[:,0]
predictions_for_false = model.predict(validate_heart_disease_false_x)[:,0]

true_positives = np.sum(np.rint(predictions_for_true) == np.ones_like(predictions_for_true)).tolist()
true_negatives = np.sum(np.rint(predictions_for_false) == np.zeros_like(predictions_for_false)).tolist()
false_positives = len(predictions_for_false)-true_negatives
false_negatives = len(predictions_for_true)-true_positives


current_datetime = datetime.datetime.now(timezone.utc)
metrics = {"true_positives": true_positives, "true_negatives": true_negatives, "false_positives": false_positives, "false_negatives" : false_negatives, "datetime_utc" : str(current_datetime)}
history = []
try:
    with open("metrics.json","r") as f:
       history = json.load(f)
except FileNotFoundError:
    print('No historical metrics found')

history.append(metrics)
with open("metrics.json","w") as f:
       json.dump(history, f)


import matplotlib.pyplot as plt
true_positives_history = [x["true_positives"] for x in history]
true_negatives_history = [x["true_negatives"] for x in history]
false_positives_history = [x["false_positives"] for x in history]
false_negatives_history = [x["false_negatives"] for x in history]

plt.plot(true_positives_history, "-go")
plt.plot(true_negatives_history, "-bo")
plt.plot(false_positives_history, "-ro")
plt.plot(false_negatives_history, "-mo")

plt.legend(["True positives", "True negatives", "False positives", "False negatives"])
plt.xlabel("Build number")
plt.ylabel("Metric value")
plt.title("Model evaluation history")
plt.savefig("metrics.jpg")