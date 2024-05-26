import pandas as pd
import os
import zipfile
with zipfile.ZipFile("dataset_cleaned.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset_cleaned_extracted")


train = pd.read_csv(os.path.join("dataset_cleaned_extracted","train.csv"))
test = pd.read_csv(os.path.join("dataset_cleaned_extracted","test.csv"))
valid = pd.read_csv(os.path.join("dataset_cleaned_extracted","valid.csv"))


num_columns = train.select_dtypes(['float64']).columns

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

train_x = train[x_columns]
train_y = train[y_column]

test_x = test[x_columns]
test_y = test[y_column]

train.info()

import os

env_var_list = ["INPUT_VERBOSE", "INPUT_EPOCHS", "INPUT_LEARNING_RATE", "INPUT_PATIENCE"]

if not all(env_var in os.environ for env_var in env_var_list):
    print("Nie znaleziono parametrów do treningu w zmiennych środowiskowych")
    parameter_epochs = 11
    parameter_patience = 3
    parameter_learning_rate = 0.001
    parameter_verbose = 2
else:
    parameter_epochs = int(os.environ.get('INPUT_EPOCHS'))
    parameter_patience = int(os.environ.get('INPUT_PATIENCE'))
    parameter_learning_rate = float(os.environ.get('INPUT_LEARNING_RATE'))
    parameter_verbose = int(os.environ.get('INPUT_VERBOSE'))

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam


def create_model():
    inputs = keras.Input(shape=(35,))
    dense1 = layers.Dense(64, activation="relu")(inputs)
    dropout1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(32, activation="relu")(dropout1)
    dropout2 = layers.Dropout(0.2)(dense2)
    output = layers.Dense(1, activation="sigmoid")(dropout2)
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=parameter_learning_rate),
                  metrics=['accuracy'])
    return model


model = create_model()

model.summary()

# Early stopping dla regularyzacji
callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=parameter_patience,
                                         restore_best_weights=True)

history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=parameter_epochs, callbacks=[callback],
                    verbose=parameter_verbose)

model.save("model.keras")

valid_x = valid[x_columns]
valid_y = valid[y_column]

import numpy as np

predictions = model.predict(valid_x)[:, 0]
true_answers = valid_y.to_numpy()
validation_accuracy = np.sum(np.rint(predictions) == true_answers) / len(true_answers)
print(f"Poprawność na zbiorze walidacyjnym: {validation_accuracy:.2%}")
print("Przykładowe predykcje (surowe):")
print(predictions[:100])
print("Przykładowe predykcje (zaokrąglone):")
print(np.rint(predictions)[:100])
print("Prawdziwe wartości:")
print(true_answers[:100])
