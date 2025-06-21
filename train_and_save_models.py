import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Train and save SVM
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, 'svm_model.pkl')

# Train and save Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)
joblib.dump(dt, 'dt_model.pkl')

# Train and save KNN
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
joblib.dump(knn, 'knn_model.pkl')
