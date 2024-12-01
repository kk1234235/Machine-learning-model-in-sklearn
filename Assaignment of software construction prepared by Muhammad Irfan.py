#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier


# # Decision Tree

# In[19]:


# Define dataset path
dataset_dir = r"D:/machine learning files/archive (2)/Dataset"  # Replace with the actual path

# Initialize variables
images = []
labels = []

# Map classes to numeric labels
class_names = os.listdir(dataset_dir)  # ['class_1', 'class_2', 'class_3', 'class_4']
class_labels = {name: idx for idx, name in enumerate(class_names)}

# Load images and labels
for class_name, class_idx in class_labels.items():
    class_path = os.path.join(dataset_dir, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # Load and preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (256, 256))  # Resize to 64x64
        images.append(img.flatten())  # Flatten image into a 1D vector
        labels.append(class_idx)

# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the images
images = images / 255.0  # Scale pixel values to [0, 1]


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# In[4]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[5]:


y_pred = model.predict(X_test)


# In[6]:


# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_pred,y_test)
print(f"accuracy: {accuracy}")


# # KNN

# In[7]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[8]:


y_pred = knn.predict(X_test)


# In[9]:


# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_pred,y_test)
print(f"accuracy: {accuracy}")


# # Random Forest

# In[10]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_model.predict(X_test)


# In[11]:


# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_pred,y_test)
print(f"accuracy: {accuracy}")


# # SVM (Support Vector Machine)

# In[12]:


# Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)  # Linear kernel for simplicity
svm_model.fit(X_train, y_train)

# Predict on the test data
y_pred = svm_model.predict(X_test)


# In[13]:


# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_pred,y_test)
print(f"accuracy: {accuracy}")


# # Naive Bais

# In[14]:


# Initialize the Gaussian Naive Bayes classifier
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predict on the test data
y_pred = nb_model.predict(X_test)


# In[15]:


# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_pred,y_test)
print(f"accuracy: {accuracy}")


# # Ada boost

# In[16]:


# Define the base learner (Decision Tree Stump)
base_learner = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoost classifier
adaboost_model = AdaBoostClassifier(base_estimator=base_learner, n_estimators=50, random_state=42)

# Train the AdaBoost model
adaboost_model.fit(X_train, y_train)

# Predict on the test data
y_pred = adaboost_model.predict(X_test)


# In[17]:


# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_pred,y_test)
print(f"accuracy: {accuracy}")


# # Xg Boost

# In[21]:


# Initialize the XGBoost Classifier
xgb_model = XGBClassifier(
    objective='multi:softmax',  # For multi-class classification
    num_class=4,               # Number of classes
    n_estimators=100,          # Number of boosting rounds
    learning_rate=0.1,         # Learning rate
    max_depth=6,               # Maximum depth of trees
    random_state=42
)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Predict on the test data
y_pred = xgb_model.predict(X_test)


# In[22]:


# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
accuracy = accuracy_score(y_pred,y_test)
print(f"accuracy: {accuracy}")



# In[ ]:




