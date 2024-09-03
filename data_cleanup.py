### 1. Download the dataset
import zipfile
with zipfile.ZipFile("personal-key-indicators-of-heart-disease.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset_extracted")
import pandas as pd
# Opening the dataset containing NaNs and manually cleaning it up for practice
df = pd.read_csv("dataset_extracted/2022/heart_2022_with_nans.csv")
## View the uncleaned dataset
df.info()
df.head()
df.describe()
df["HadHeartAttack"].value_counts().plot(kind="pie")
df["HadHeartAttack"].value_counts()

## 2. Dividing the dataset (train / dev / test - 8:1:1)) and oversampling
from sklearn.model_selection import train_test_split
# Use train_test_split twice to divide into 3 datasets
train, test_and_valid = train_test_split(df, test_size=0.2) #0.8 train, 0.2 test&valid

test, valid = train_test_split(test_and_valid, test_size=0.5) #0.1 test, 0.1 valid
train["HadHeartAttack"].value_counts()
def oversample(dataset):
    num_true = len(dataset[dataset["HadHeartAttack"]=="Yes"])
    num_false = len(dataset[dataset["HadHeartAttack"]=="No"])
    num_oversampling_steps = num_false//num_true
    oversampled = dataset.copy()
    for x in range(num_oversampling_steps):
        oversampled = pd.concat([oversampled, dataset[dataset["HadHeartAttack"]=="Yes"]], ignore_index=True)
    return oversampled
train = oversample(train)
train["HadHeartAttack"].value_counts().plot(kind="pie")
test["HadHeartAttack"].value_counts().plot(kind="pie")
valid["HadHeartAttack"].value_counts().plot(kind="pie")
df["SmokerStatus"].value_counts().plot(kind="pie")
df["ECigaretteUsage"].value_counts().plot(kind="pie")
df["CovidPos"].value_counts().plot(kind="pie")
## 3.1 normalization pt 1 : converting text to numerical and categorical data
df["Sex"].unique()
df["GeneralHealth"].unique()
health_map = {
    "Excellent": 5,
    "Very good": 4,
    "Good": 3,
    "Fair": 2,
    "Poor": 1
}
for col in df:
    print(f"{col}:")
    print(df[col].unique())
from collections import defaultdict
def normalize_dataset(dataset):
    dataset["GeneralHealth"] = dataset["GeneralHealth"].map(defaultdict(lambda: float('NaN'), health_map), na_action='ignore')
    dataset["Sex"] = dataset["Sex"].map({"Female":0,"Male":1}).astype(float) # Convert text to numerical data
    dataset.rename(columns ={"Sex":"Male"},inplace=True)
    dataset["State"] = dataset["State"].astype('category')
    dataset["PhysicalHealthDays"].astype(float)
    dataset["MentalHealthDays"].astype(float)
    dataset["LastCheckupTime"] = dataset["LastCheckupTime"].fillna("Unknown").astype('category') # I use fillna --> median later, but that doesn't work on categorical columns, so I'm doing this before converting.
    dataset["PhysicalActivities"]= dataset["PhysicalActivities"].map({"No":0,"Yes":1})
    dataset["SleepHours"].astype(float)
    dataset["RemovedTeeth"] = dataset["RemovedTeeth"].map(defaultdict(lambda: float('NaN'), {"None of them":0,"1 to 5":1, "6 or more, but not all":2, "All":3}), na_action='ignore')
    dataset["HadHeartAttack"]= dataset["HadHeartAttack"].map({"No":0,"Yes":1})
    dataset["HadAngina"]= dataset["HadAngina"].map({"No":0,"Yes":1})
    dataset["HadStroke"]= dataset["HadStroke"].map({"No":0,"Yes":1})
    dataset["HadAsthma"]= dataset["HadAsthma"].map({"No":0,"Yes":1})
    dataset["HadSkinCancer"]= dataset["HadSkinCancer"].map({"No":0,"Yes":1})
    dataset["HadCOPD"]= dataset["HadCOPD"].map({"No":0,"Yes":1})
    dataset["HadDepressiveDisorder"]= dataset["HadDepressiveDisorder"].map({"No":0,"Yes":1})
    dataset["HadKidneyDisease"]= dataset["HadKidneyDisease"].map({"No":0,"Yes":1})
    dataset["HadArthritis"]= dataset["HadArthritis"].map({"No":0,"Yes":1})
    dataset["HadDiabetes"]= dataset["HadDiabetes"].map({"No":0,"Yes, but only during pregnancy (female)":1,"No, pre-diabetes or borderline diabetes":2,"Yes":3})

    dataset["DeafOrHardOfHearing"]= dataset["DeafOrHardOfHearing"].map({"No":0,"Yes":1})
    dataset["BlindOrVisionDifficulty"]= dataset["BlindOrVisionDifficulty"].map({"No":0,"Yes":1})
    dataset["DifficultyConcentrating"]= dataset["DifficultyConcentrating"].map({"No":0,"Yes":1})
    dataset["DifficultyWalking"]= dataset["DifficultyWalking"].map({"No":0,"Yes":1})
    dataset["DifficultyDressingBathing"]= dataset["DifficultyDressingBathing"].map({"No":0,"Yes":1})
    dataset["DifficultyErrands"]= dataset["DifficultyErrands"].map({"No":0,"Yes":1})
    dataset["SmokerStatus"]= dataset["SmokerStatus"].map({"Never smoked":0,"Current smoker - now smokes some days":1,"Former smoker":2,"Current smoker - now smokes every day":3})
    dataset["ECigaretteUsage"]= dataset["ECigaretteUsage"].map({"Never used e-cigarettes in my entire life":0,"Not at all (right now)":1,"Use them some days":2,"Use them every day":3})
    dataset["ChestScan"]= dataset["ChestScan"].map({"No":0,"Yes":1})
    dataset["RaceEthnicityCategory"] = dataset["RaceEthnicityCategory"].fillna("Unknown").astype('category')
    dataset["AgeCategory"] = dataset["AgeCategory"].fillna("Unknown").astype('category')
    dataset["HeightInMeters"] = dataset["HeightInMeters"].astype(float)
    dataset["WeightInKilograms"] = dataset["WeightInKilograms"].astype(float)
    dataset["BMI"] = dataset["BMI"].astype(float)
    dataset["AlcoholDrinkers"]= dataset["AlcoholDrinkers"].map({"No":0,"Yes":1})
    dataset["HIVTesting"]= dataset["HIVTesting"].map({"No":0,"Yes":1})
    dataset["FluVaxLast12"]= dataset["FluVaxLast12"].map({"No":0,"Yes":1})
    dataset["PneumoVaxEver"]= dataset["PneumoVaxEver"].map({"No":0,"Yes":1})
    dataset["TetanusLast10Tdap"]= dataset["TetanusLast10Tdap"].apply(lambda x: float('NaN') if type(x)!=str else 1.0 if 'Yes,' in x else 1.0 if 'No,' in x else float('NaN'))
    dataset["HighRiskLastYear"]= dataset["HighRiskLastYear"].map({"No":0,"Yes":1})
    dataset["CovidPos"]= dataset["CovidPos"].map({"No":0,"Yes":1})
test.head()
normalize_dataset(test)
test.head()
test.info()
normalize_dataset(train)
normalize_dataset(valid)
train.describe()
test.describe()
valid.describe()
import seaborn as sns
sns.set_theme()
g = sns.catplot(
    data=train, kind="bar",
    x="GeneralHealth", y="WeightInKilograms", hue="HadHeartAttack",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("General health index", "Body mass (kg)")
g.legend.set_title("Had heart attack")
valid.groupby('SmokerStatus', as_index=False)['HadHeartAttack'].mean()
valid.groupby('GeneralHealth', as_index=False)['HadHeartAttack'].mean()
valid.pivot_table('HadHeartAttack',index='GeneralHealth', columns='SmokerStatus')
## 3.2 Normalization pt 2 - scaling numerical data to the range 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
def scale_float_columns(dataset):
    numerical_columns = list(dataset.select_dtypes(include=['float64']).columns)
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])
test.head()
scale_float_columns(test)
scale_float_columns(train)
scale_float_columns(valid)
test.head()
## 4. Cleaning up missing values
print(df.shape[0])
print(df.shape[0] - df.dropna().shape[0])
test.head()

numeric_columns = train.select_dtypes(include=['number']).columns
test[numeric_columns] = test[numeric_columns].fillna(test[numeric_columns].median().iloc[0])
train[numeric_columns] = train[numeric_columns].fillna(train[numeric_columns].median().iloc[0])
valid[numeric_columns] = valid[numeric_columns].fillna(valid[numeric_columns].median().iloc[0])

test.head()
test["HighRiskLastYear"].value_counts()
test["HighRiskLastYear"].isna().sum()
test.info()
train.info()
valid.info()

cat_columns = test.select_dtypes(['category']).columns

test.to_csv("test.csv")
train.to_csv("train.csv")
valid.to_csv("valid.csv")