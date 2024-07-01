import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Paths to train and test directories
train_dir = 'Train'
test_dir = 'Test'

# Function to load images and labels for specific letters
def load_images_for_letters(folder, letters):
    images = []
    labels = []
    for letter in letters:
        letter_folder = os.path.join(folder, letter)
        if os.path.isdir(letter_folder):
            for file in os.listdir(letter_folder):
                img_path = os.path.join(letter_folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(letter)
    return images, labels

# Load training and testing data for letters 'A' and 'B'
train_images, train_labels = load_images_for_letters(train_dir, ['A', 'B'])
test_images, test_labels = load_images_for_letters(test_dir, ['A', 'B'])

# Function to flatten images
def flatten_images(images):
    flattened_images = []
    for img in images:
        flattened_images.append(img.flatten())
    return np.array(flattened_images)

# Flatten training and testing images
train_images_flattened = flatten_images(train_images)
test_images_flattened = flatten_images(test_images)

# Apply PCA to reduce dimensions
pca = PCA(n_components=50)  # You can adjust the number of components
train_features_pca = pca.fit_transform(train_images_flattened)
test_features_pca = pca.transform(test_images_flattened)

# Convert labels to numerical format
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Train the SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(train_features_pca, train_labels_encoded)

# Predict on the test set
test_predictions = clf.predict(test_features_pca)

# Evaluate the classifier
accuracy = accuracy_score(test_labels_encoded, test_predictions)
report = classification_report(test_labels_encoded, test_predictions, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)





