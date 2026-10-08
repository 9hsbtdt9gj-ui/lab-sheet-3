# ==============================================================================
# COER UNIVERSITY - LAB SHEET-03
# Supervised Learning (Regression Models)
# ==============================================================================

# ------------------------------------------------------------------------------
# PROGRAM 1: Load a Regression Dataset using Pandas
# ------------------------------------------------------------------------------
import pandas as pd

# Load sample regression dataset
df = pd.read_csv("housing.csv")


# ------------------------------------------------------------------------------
# PROGRAM 2: Display First and Last Five Records
# ------------------------------------------------------------------------------
print("First 5 records:")
df.head()

print("Last 5 records:")
df.tail()


# ------------------------------------------------------------------------------
# PROGRAM 3: Explore Dataset Information and Descriptive Statistics
# ------------------------------------------------------------------------------
df.info()
df.describe()


# ------------------------------------------------------------------------------
# PROGRAM 4: Identify Input (Independent) and Output (Dependent) Variables
# ------------------------------------------------------------------------------
# Assuming 'SquareFeet' and 'Bedrooms' are features, and 'Price' is the target variable
X = df[["SquareFeet", "Bedrooms"]]
y = df["Price"]


# ------------------------------------------------------------------------------
# PROGRAM 5: Split Dataset into Training and Testing Sets
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------------------------------------------------------
# PROGRAM 6: Implement Simple Linear Regression Model
# ------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

simple_lr_model = LinearRegression()


# ------------------------------------------------------------------------------
# PROGRAM 7: Train Simple Linear Regression Model
# ------------------------------------------------------------------------------
X_train_simple = X_train[["SquareFeet"]]
X_test_simple = X_test[["SquareFeet"]]

simple_lr_model.fit(X_train_simple, y_train)


# ------------------------------------------------------------------------------
# PROGRAM 8: Predict Output Values using Trained Linear Regression Model
# ------------------------------------------------------------------------------
y_pred_simple = simple_lr_model.predict(X_test_simple)


# ------------------------------------------------------------------------------
# PROGRAM 9: Visualize Linear Regression Line using Matplotlib
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.scatter(X_test_simple, y_test, color="blue", label="Actual")
plt.plot(
    X_test_simple, y_pred_simple, color="red", linewidth=2, label="Regression Line"
)
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 10: Compare Actual and Predicted Values
# ------------------------------------------------------------------------------
comparison_simple_df = pd.DataFrame(
    {"Actual": y_test, "Predicted": y_pred_simple}
)


# ------------------------------------------------------------------------------
# PROGRAM 11: Display Regression Coefficient and Intercept
# ------------------------------------------------------------------------------
print("Coefficient:", simple_lr_model.coef_[0])
print("Intercept:", simple_lr_model.intercept_)


# ------------------------------------------------------------------------------
# PROGRAM 12: Predict Output for New User-Defined Input Values
# ------------------------------------------------------------------------------
import numpy as np

new_input = np.array([[2500]])
new_prediction = simple_lr_model.predict(new_input)


# ------------------------------------------------------------------------------
# PROGRAM 13: Implement Multiple Linear Regression
# ------------------------------------------------------------------------------
multiple_lr_model = LinearRegression()


# ------------------------------------------------------------------------------
# PROGRAM 14: Train Multiple Linear Regression Model
# ------------------------------------------------------------------------------
multiple_lr_model.fit(X_train, y_train)


# ------------------------------------------------------------------------------
# PROGRAM 15: Predict Output Values using Testing Dataset
# ------------------------------------------------------------------------------
y_pred_multi = multiple_lr_model.predict(X_test)


# ------------------------------------------------------------------------------
# PROGRAM 16: Compare Actual and Predicted Values Graphically
# ------------------------------------------------------------------------------
plt.scatter(y_test, y_pred_multi, color="green")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2,
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Multiple Regression)")
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 17: Analyze Effect of Each Independent Variable on Prediction
# ------------------------------------------------------------------------------
coefficients = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": multiple_lr_model.coef_}
)


# ------------------------------------------------------------------------------
# PROGRAM 18: Implement Polynomial Regression of Degree 2
# ------------------------------------------------------------------------------
from sklearn.preprocessing import PolynomialFeatures

poly_features_deg2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly_features_deg2.fit_transform(X_train_simple)
X_test_poly2 = poly_features_deg2.transform(X_test_simple)

poly_model_deg2 = LinearRegression()
poly_model_deg2.fit(X_train_poly2, y_train)


# ------------------------------------------------------------------------------
# PROGRAM 19: Implement Polynomial Regression of Degree 3
# ------------------------------------------------------------------------------
poly_features_deg3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly_features_deg3.fit_transform(X_train_simple)
X_test_poly3 = poly_features_deg3.transform(X_test_simple)

poly_model_deg3 = LinearRegression()
poly_model_deg3.fit(X_train_poly3, y_train)


# ------------------------------------------------------------------------------
# PROGRAM 20: Compare Linear Regression and Polynomial Regression Models
# ------------------------------------------------------------------------------
y_pred_poly2 = poly_model_deg2.predict(X_test_poly2)
y_pred_poly3 = poly_model_deg3.predict(X_test_poly3)


# ------------------------------------------------------------------------------
# PROGRAM 21: Visualize Polynomial Regression Curves
# ------------------------------------------------------------------------------
X_grid = np.linspace(
    X_train_simple.min().values[0], X_train_simple.max().values[0], 100
).reshape(-1, 1)

plt.scatter(X_test_simple, y_test, color="blue", label="Actual Data")
plt.plot(
    X_grid,
    poly_model_deg2.predict(poly_features_deg2.transform(X_grid)),
    color="orange",
    label="Degree 2 Curve",
)
plt.plot(
    X_grid,
    poly_model_deg3.predict(poly_features_deg3.transform(X_grid)),
    color="red",
    label="Degree 3 Curve",
)
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Polynomial Regression Curves")
plt.legend()
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 22: Predict Output Values using Polynomial Regression Model
# ------------------------------------------------------------------------------
poly_predictions = poly_model_deg2.predict(X_test_poly2)


# ------------------------------------------------------------------------------
# PROGRAM 23: Compare Prediction Accuracy for Different Polynomial Degrees
# ------------------------------------------------------------------------------
from sklearn.metrics import r2_score

r2_deg2 = r2_score(y_test, y_pred_poly2)
r2_deg3 = r2_score(y_test, y_pred_poly3)


# ------------------------------------------------------------------------------
# PROGRAM 24: Calculate Mean Absolute Error (MAE)
# ------------------------------------------------------------------------------
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred_multi)


# ------------------------------------------------------------------------------
# PROGRAM 25: Calculate Mean Squared Error (MSE)
# ------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred_multi)


# ------------------------------------------------------------------------------
# PROGRAM 26: Calculate Root Mean Squared Error (RMSE)
# ------------------------------------------------------------------------------
rmse = np.sqrt(mse)


# ------------------------------------------------------------------------------
# PROGRAM 27: Calculate R-squared (R2) Score
# ------------------------------------------------------------------------------
r2 = r2_score(y_test, y_pred_multi)


# ------------------------------------------------------------------------------
# PROGRAM 28: Compare Performance using Evaluation Metrics
# ------------------------------------------------------------------------------
metrics_comparison = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Polynomial (Deg 2)"],
        "MAE": [
            mean_absolute_error(y_test, y_pred_simple),
            mean_absolute_error(y_test, y_pred_poly2),
        ],
        "MSE": [
            mean_squared_error(y_test, y_pred_simple),
            mean_squared_error(y_test, y_pred_poly2),
        ],
        "R2 Score": [
            r2_score(y_test, y_pred_simple),
            r2_score(y_test, y_pred_poly2),
        ],
    }
)


# ------------------------------------------------------------------------------
# PROGRAM 29: Interpret Meaning of MSE and R2 Values
# ------------------------------------------------------------------------------
def interpret_metrics(mse_val, r2_val):
    interpretation = f"MSE indicates average squared error ({mse_val:.2f}). "
    interpretation += f"R2 score ({r2_val:.2f}) indicates proportion of variance explained by the model."
    return interpretation


# ------------------------------------------------------------------------------
# PROGRAM 30: Visualize Prediction Errors using Scatter Plots
# ------------------------------------------------------------------------------
errors = y_test - y_pred_multi
plt.scatter(y_pred_multi, errors, color="purple")
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Errors)")
plt.title("Residual / Prediction Error Plot")
plt.show()


# ------------------------------------------------------------------------------
# PROGRAM 31: Train Regression Model using Standardized Features
# ------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_lr_model = LinearRegression()
scaled_lr_model.fit(X_train_scaled, y_train)


# ------------------------------------------------------------------------------
# PROGRAM 32: Compare Model Performance Before and After Feature Scaling
# ------------------------------------------------------------------------------
y_pred_scaled = scaled_lr_model.predict(X_test_scaled)
r2_unscaled = r2_score(y_test, y_pred_multi)
r2_scaled = r2_score(y_test, y_pred_scaled)


# ------------------------------------------------------------------------------
# PROGRAM 33: Perform Regression using Another Real-World Dataset
# ------------------------------------------------------------------------------
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing(as_frame=True)
df_cal = california.frame

X_cal = df_cal.drop(columns=["MedHouseVal"])
y_cal = df_cal["MedHouseVal"]

X_tr_cal, X_te_cal, y_tr_cal, y_te_cal = train_test_split(
    X_cal, y_cal, test_size=0.2, random_state=42
)

cal_model = LinearRegression()
cal_model.fit(X_tr_cal, y_tr_cal)


# ------------------------------------------------------------------------------
# PROGRAM 34: Save Trained Regression Model using Joblib
# ------------------------------------------------------------------------------
import joblib

joblib.dump(multiple_lr_model, "multiple_lr_model.pkl")


# ------------------------------------------------------------------------------
# PROGRAM 35: Load Saved Model and Predict New Data
# ------------------------------------------------------------------------------
loaded_model = joblib.load("multiple_lr_model.pkl")
new_sample = X_test.iloc[[0]]
new_data_prediction = loaded_model.predict(new_sample)
