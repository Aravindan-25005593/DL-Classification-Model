# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
<img width="915" height="407" alt="image" src="https://github.com/user-attachments/assets/082c22b8-2599-4118-b9cd-b26bce94540a" />


## DESIGN STEPS
### STEP 1:
Load the Iris dataset using a suitable library.

### STEP 2:
Preprocess the data by handling missing values and normalizing features.

### STEP 3:
Split the dataset into training and testing sets.

### STEP 4:
Train a classification model using the training data.

### STEP 5:
Evaluate the model on the test data and calculate accuracy.

### STEP 6:
Display the test accuracy, confusion matrix, and classification report. 




## PROGRAM

### Name: ARAVINDAN SD

### Register Number: 212224243001

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Step 1: Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Convert to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Step 3: Display First and Last 5 Rows
print("First 5 rows of dataset:\n", df.head())
print("\nLast 5 rows of dataset:\n", df.tail())

# Step 4: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Convert to PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 7: Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16
)

# Step 8: Define Neural Network Model
class IrisClassifier(nn.Module):

    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Step 9: Training Function
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Loss: {loss.item():.4f}"
            )


# Step 10: Initialize Model, Loss Function and Optimizer
model = IrisClassifier(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 11: Train the Model
train_model(
    model, train_loader, criterion, optimizer, epochs=100
)

# Step 12: Evaluate the Model
model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Step 13: Compute Evaluation Metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)

class_report = classification_report(
    actuals,
    predictions,
    target_names=iris.target_names
)

# Print Results
print("\nName: ARAVINDAN SD")
print("Register No: 212224243001")
print(f"Test Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Step 14: Plot Confusion Matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    fmt="g"
)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Step 15: Make a Sample Prediction
sample_input = X_test[5].unsqueeze(0)

with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = iris.target_names[predicted_class_index]

print("\nName: ARAVINDAN SD")
print("Register No: 212224243001")
print(f"Predicted class for sample input: {predicted_class_label}")
print(f"Actual class for sample input: {iris.target_names[y_test[5].item()]}")
```

### Dataset Information

<img width="652" height="513" alt="image" src="https://github.com/user-attachments/assets/97a1560d-4556-49c1-9f4a-c824d9f84f0d" />


### OUTPUT

## Confusion Matrix

<img width="531" height="755" alt="image" src="https://github.com/user-attachments/assets/e547bf3a-87eb-44cd-b75e-e69f6b30c294" />


## Classification Report

<img width="262" height="192" alt="image" src="https://github.com/user-attachments/assets/801e8f9d-a150-46ae-85c8-ebb6aab221ec" />


### New Sample Data Prediction

<img width="386" height="85" alt="image" src="https://github.com/user-attachments/assets/02b06dc3-fbae-4590-ad8b-e9619341c182" />


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch
