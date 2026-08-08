import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error

df = pd.read_csv("student_performance/data/student_performance_data.csv")
print(df.head())
print("Null entries in Study Hours is " , df['Study Hours'].isnull().sum())
print("Null entries in Attendance is ", df['Attendance (%)'].isnull().sum())
print("Null entries in Grade is  ", df['FinalGrade'].isnull().sum())

print("Number of incorrect entries in Attendance is ", ((df['Attendance (%)']< 0 ) |(df['Attendance (%)']> 100 )) .sum())
print("Number of incorrect entries in Study Hours is ", ((df['Study Hours']<0) .sum()))
print((df[df['Study Hours']<0])) #there are 10 rows with incorrect data of both attributes, so we drop
bad_rows = ((df['Study Hours'] < 0) |(df['Attendance (%)'] < 0) |(df['Attendance (%)'] > 100))
# Remove identifier columns that have no predictive value
df.drop(columns=['StudentID', 'Name','AttendanceRate'], inplace=True)
df = df[~bad_rows]
# Drop rows where target (FinalGrade) is missing
df = df.dropna(subset=['FinalGrade'])
#fill missing data
df['Study Hours'] = df['Study Hours'].fillna(df['Study Hours'].median())
df['StudyHoursPerWeek'] = df['StudyHoursPerWeek'].fillna(df['StudyHoursPerWeek'].median())
df['Attendance (%)'] = df['Attendance (%)'].fillna(df['Attendance (%)'].median())
df['PreviousGrade'] = df['PreviousGrade'].fillna(df['PreviousGrade'].median())
df['ExtracurricularActivities'] = df['ExtracurricularActivities'].fillna(df['ExtracurricularActivities'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['ParentalSupport'] = df['ParentalSupport'].fillna(df['ParentalSupport'].mode()[0])
df['Online Classes Taken'] = df['Online Classes Taken'].fillna(df['Online Classes Taken'].mode()[0])
#easier encoding as only 2 categories 
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Online Classes Taken'] = df['Online Classes Taken'].map({False: 0, True: 1})
df['ParentalSupport'] = df['ParentalSupport'].map({'Low': 0,'Medium': 1,'High': 2})
#Train test split
y = df['FinalGrade']
X = df.drop(columns= 'FinalGrade')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Metrics
print("MSE:", mean_squared_error (y_test, y_pred))
print("MAE:", mean_absolute_error( y_test, y_pred))
print("R²:", r2_score( y_test, y_pred))
#There is less relation , graphs show that

sns.scatterplot(data=df, x='Study Hours', y='FinalGrade')
plt.show()
sns.scatterplot(data=df, x='Attendance (%)', y='FinalGrade')
plt.show()
sns.scatterplot(data=df, x='PreviousGrade', y='FinalGrade')
plt.show()