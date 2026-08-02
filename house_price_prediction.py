import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("Housing.csv")

# Display the first 5 rows
print("===== FIRST 5 ROWS =====")
print(data.head())
# Display dataset information
print("\n===== DATASET INFORMATION =====")
print(data.info())

# Display dataset shape
print("\n===== DATASET SHAPE =====")
print(data.shape)

# Display column names
print("\n===== COLUMN NAMES =====")
print(data.columns)
# Check for missing values
print("\n===== MISSING VALUES =====")
print(data.isnull().sum())
# Convert categorical columns into numerical values
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})

# Convert furnishing status into numerical values
data['furnishingstatus'] = data['furnishingstatus'].map({
    'furnished': 0,
    'semi-furnished': 1,
    'unfurnished': 2
})

# Display the first 5 rows after preprocessing
print("\n===== DATA AFTER PREPROCESSING =====")
print(data.head())
# Separate features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Display the shape of features and target
print("\n===== FEATURES AND TARGET =====")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display the shape of training and testing data
print("\n===== TRAINING AND TESTING DATA =====")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# Create the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print("\n===== MODEL TRAINING COMPLETED =====")
print("Linear Regression model has been trained successfully.")
# Make predictions using the test data
y_pred = model.predict(X_test)

# Display the first 5 predicted prices
print("\n===== PREDICTED HOUSE PRICES =====")
print(y_pred[:5])
# Calculate model evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
print("\n===== MODEL EVALUATION =====")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
# Plot actual vs predicted house prices
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.show()
# Predict the price of a new house using user input

print("\n===== ENTER NEW HOUSE DETAILS =====")

area = float(input("Enter area of the house: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
stories = int(input("Enter number of stories: "))

mainroad = int(input("Is the house connected to the main road? (1 = Yes, 0 = No): "))
guestroom = int(input("Does the house have a guestroom? (1 = Yes, 0 = No): "))
basement = int(input("Does the house have a basement? (1 = Yes, 0 = No): "))
hotwaterheating = int(input("Does the house have hot water heating? (1 = Yes, 0 = No): "))
airconditioning = int(input("Does the house have air conditioning? (1 = Yes, 0 = No): "))

parking = int(input("Enter number of parking spaces: "))
prefarea = int(input("Is the house in a preferred area? (1 = Yes, 0 = No): "))

furnishingstatus = int(input(
    "Enter furnishing status (0 = Furnished, 1 = Semi-furnished, 2 = Unfurnished): "
))

# Create input data for the model
new_house = np.array([[
    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad,
    guestroom,
    basement,
    hotwaterheating,
    airconditioning,
    parking,
    prefarea,
    furnishingstatus
]])

# Predict house price
predicted_price = model.predict(new_house)

# Display predicted price
print("\n===== NEW HOUSE PRICE PREDICTION =====")
print("Predicted House Price: ₹", round(predicted_price[0], 2))