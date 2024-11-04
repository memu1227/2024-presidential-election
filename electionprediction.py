import polls
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Drop rows with missing values in 'pct' and 'methodology'
poll_data = polls.poll_data.dropna(subset=['pct', 'methodology','sample_size'])

# Define target variable: 1 for Kamala Harris, 0 for Donald Trump
poll_data['winner'] = poll_data['candidate_name'].apply(lambda x: 1 if x == 'Kamala Harris' else 0)

# Select features including pct and others
features = ['pct', 'sample_size', 'methodology', 'state']
X = poll_data[features]
y = poll_data['winner']

# Check for missing values in the selected features
print("Missing values before dropping:")
print(X.isnull().sum())

# Convert categorical features to numerical
X = pd.get_dummies(X, columns=['methodology', 'state'], drop_first=True)

# Check for missing values again after converting to dummies
print("Missing values after converting to dummies:")
print(X.isnull().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['pct', 'sample_size']])
X_test_scaled = scaler.transform(X_test[['pct', 'sample_size']])

# Combine scaled numerical features with categorical features
X_train_final = np.hstack((X_train_scaled, X_train.drop(columns=['pct', 'sample_size']).values))
X_test_final = np.hstack((X_test_scaled, X_test.drop(columns=['pct', 'sample_size']).values))

# Check for NaN values in final training and testing sets
print("NaN values in training set:")
print(np.isnan(X_train_final).sum())
print("NaN values in testing set:")
print(np.isnan(X_test_final).sum())

# Create and train the model
model = LogisticRegression()
model.fit(X_train_final, y_train)

# Make predictions
predictions = model.predict(X_test_final)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Print classification report for detailed performance
print(classification_report(y_test, predictions, target_names=['Donald Trump', 'Kamala Harris']))

# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Donald Trump', 'Kamala Harris'])

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png", format='png', dpi=300, bbox_inches='tight')
#plt.show()
