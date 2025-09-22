import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("D:\\Internship_project_dataset\\Vehicle_prediction\\dataset.csv")

# Drop rows where price is missing
data = data.dropna(subset=["price"])

# Drop 'description' column if it exists
if 'description' in data.columns:
    data = data.drop("description", axis=1)

# Drop any remaining rows with missing values
data = data.dropna()

# Split features and target
X = data.drop("price", axis=1)
Y = data["price"]

# Label encode categorical columns
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

Y = Y.values

# ✅ Dynamically find numeric columns
num_cols = data.select_dtypes(include=["int64", "float64"]).columns
print("Numeric columns:", num_cols)

# Heatmap for numeric features only
plt.figure(figsize=(10, 8))
sns.heatmap(data[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric features)")
plt.show()

# Other plots (these are fine)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="mileage", y="price", data=data)
plt.title("Mileage vs Price")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="fuel", y="price", data=data)
plt.title("Price vs Fuel Type")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="body", y="price", data=data)
plt.title("Body type vs Price")
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x="transmission", y="price", data=data)
plt.title("Transmission vs Price")
plt.show()

# Pairplot for numeric features only
sns.pairplot(data[num_cols])
plt.show()

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
lr_pred = lr_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, Y_train)
dt_pred = dt_model.predict(X_test)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("Mean Squared Error (MSE):", mean_squared_error(y_true, y_pred))
    print("R-squared (R²):", r2_score(y_true, y_pred))
    print()

# Evaluate all models
evaluate_model("Random Forest Regressor", Y_test, rf_pred)
evaluate_model("Linear Regression", Y_test, lr_pred)
evaluate_model("Decision Tree Regressor", Y_test, dt_pred)

# Save Random Forest model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
