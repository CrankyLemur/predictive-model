# This is a basic random forest predictive model that aims to predict
# employee attrition.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import compute_class_weight

# Import dataset
df = pd.read_csv("./case_study.csv")

# Filter training dataset to train using high performer traits
df = df.loc[df["PerformanceRating"] > 3, :]

# Split the dataset into training and testing sets
X = df[
    [
        # "Age",
        # "Gender",
        "MonthlyIncome",
        # "Department",
        # "NumCompaniesWorked",
        # "workingfromhome",
        # "BusinessTravel",
        # "DistanceFromHome",
        "JobSatisfaction",
        # "complaintfiled",
        "PercentSalaryHike",
        "TotalWorkingYears",
        "YearsAtCompany",
        # "YearsSinceLastPromotion",
    ]
]
y = df["Left"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Calculate class weights
class_weights = compute_class_weight("balanced", classes=[0, 1], y=y_train)

# Train a random forest model with class weights
rf = RandomForestClassifier(
    n_estimators=10000,
    max_depth=2,
    min_samples_split=10,
    class_weight={0: class_weights[0], 1: class_weights[1]},
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate the performance of the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Confusion matrix:\n", conf_matrix)
print("Classification report:\n", class_report)

# Predict the probability of leaving for a new employee
employee1 = [
    1,  # income category
    1,  # job satisfaction
    11,  # salary hike
    1,  # working years
    1,  # years at company
]
risk_score = rf.predict_proba([employee1])[0][1]  # probability of leaving
print("The probability of this employee leaving the company is:", risk_score)
