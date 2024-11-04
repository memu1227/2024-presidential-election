import polls
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Drop rows with missing values in 'pct', 'sample_size', and 'state'
poll_data = polls.poll_data.dropna(subset=['pct', 'sample_size', 'state'])

# Define target variable: 1 for Kamala Harris, 0 for Donald Trump
poll_data['winner'] = poll_data['candidate_name'].apply(lambda x: 1 if x == 'Kamala Harris' else 0)

# Select features excluding methodology
features = ['pct', 'sample_size', 'state']
X = poll_data[features]
y = poll_data['winner']

# Convert categorical features to numerical
X = pd.get_dummies(X, columns=['state'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['pct', 'sample_size']])
X_test_scaled = scaler.transform(X_test[['pct', 'sample_size']])

# Combine scaled numerical features with categorical features
X_train_final = np.hstack((X_train_scaled, X_train.drop(columns=['pct', 'sample_size']).values))
X_test_final = np.hstack((X_test_scaled, X_test.drop(columns=['pct', 'sample_size']).values))

# List of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Prepare a figure for confusion matrices
num_models = len(models)
fig, axes = plt.subplots(2, num_models // 2, figsize=(20, 10))  # Adjust the size as needed
axes = axes.flatten()  # Flatten the axes array for easy indexing

# Evaluate each model and plot confusion matrices
for ax, (model_name, model) in zip(axes, models.items()):
    model.fit(X_train_final, y_train)
    predictions = model.predict(X_test_final)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Donald Trump', 'Kamala Harris'])
    
    # Plot confusion matrix on the current axes
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    ax.set_title(model_name)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

# Adjust layout to avoid overlapping
plt.tight_layout(pad=3.0)  # Adjust padding between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust horizontal and vertical spacing

# Save the figure
plt.savefig("confusion_matrices_comparison.png", format='png', dpi=300, bbox_inches='tight')
plt.show()
