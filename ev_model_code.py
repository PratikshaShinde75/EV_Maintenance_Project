import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib   # ✅ ONLY joblib

# Load dataset
data = pd.read_csv(
    r"D:\EV Maintanance Project\vehicle_maintenance_data_file.csv"
)
print("✅ Dataset loaded successfully")
print(data.head())

# Encode categorical columns
data['Battery_Status_Num'] = data['Battery_Status'].map({
    'New': 0,
    'Weak': 1,
    'Old': 2
})

# Features & target
X = data[['Mileage', 'Battery_Status_Num']]
y = data['Need_Maintenance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=40,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# ✅ SAVE MODEL (joblib only)
joblib.dump(model, "ev_model.pkl", compress=3)
print("✅ EV model trained and saved as ev_model.pkl")

# Test load immediately (IMPORTANT)
test_model = joblib.load("ev_model.pkl")
print("✅ Model reload test PASSED")

# Evaluation
y_pred = model.predict(X_test)
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("🟦 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("📝 Classification Report:")
print(classification_report(y_test, y_pred))
