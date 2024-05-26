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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=int, default=2, help='Verbosity level')
parser.add_argument('--epochs', type=int, default=11, help='Number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Model learning rate')
parser.add_argument('--patience', type=int, default=3, help='Patience for the early stopping callback')


args = parser.parse_args()
parameter_epochs = args.epochs
parameter_patience = args.patience
parameter_learning_rate = args.learning_rate
parameter_verbose = args.verbose

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
