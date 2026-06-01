import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("Advertising.csv")

# Remove unwanted column
df = df.drop("Unnamed: 0", axis=1)

print("Dataset Preview:")
print(df.head())

# Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nR2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Example Prediction
tv = 200
radio = 40
newspaper = 50

prediction = model.predict([[tv, radio, newspaper]])

print("\nPredicted Sales:", prediction[0])

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Sales Prediction")
plt.show()