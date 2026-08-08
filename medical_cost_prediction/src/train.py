import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error

df = pd.read_csv ("medical_cost_prediction/data/insurance.csv")
print(df.isnull().sum()) #there are no null values 
df = pd.get_dummies(df, columns=['region'], drop_first=True) #northeast is reference 
df['smoker'] = df['smoker'].str.lower()
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df['sex'] = df['sex'].str.lower()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
print(df.head())

#model1
#Train test split
y = df['charges']
X = df.drop(columns= 'charges')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Metrics
print("MSE(with smoker):", mean_squared_error (y_test, y_pred))
print("MAE(with smoker):", mean_absolute_error( y_test, y_pred))
print("R²(with smoker):", r2_score( y_test, y_pred))
print(pd.DataFrame({'Feature': X.columns,'Coefficient': model.coef_}).sort_values('Coefficient', ascending=False))
#shows being a smoker has high impact

#actual vs predicted plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges(with smoker)")
plt.show()

#model 2
y = df['charges']
X = df.drop(columns= ['charges','smoker'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#Metrics
print("MSE(without smoker):", mean_squared_error (y_test, y_pred))
print("MAE(without smoker):", mean_absolute_error( y_test, y_pred))
print("R²(without smoker):", r2_score( y_test, y_pred))

#actual vs predicted plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges(smokers removed)")
plt.show()