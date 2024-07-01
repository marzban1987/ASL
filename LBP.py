import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.feature import local_binary_pattern

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

# Resize images to a common size
resize_dim = (64, 64)
train_images = [cv2.resize(img, resize_dim) for img in train_images]
test_images = [cv2.resize(img, resize_dim) for img in test_images]

# Function to extract LBP features
def extract_lbp_features(images, P=8, R=1):
    lbp_features = []
    for img in images:
        lbp = local_binary_pattern(img, P, R, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalize the histogram
        lbp_features.append(hist)
    return lbp_features

# Extract LBP features from training and testing images
train_features = extract_lbp_features(train_images)
test_features = extract_lbp_features(test_images)

# Convert lists to numpy arrays
train_features = np.array(train_features)
test_features = np.array(test_features)

# Ensure labels and features are of same length
train_labels = train_labels[:len(train_features)]
test_labels = test_labels[:len(test_features)]

# Convert labels to numerical format
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Standardize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Train the SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(train_features, train_labels_encoded)

# Predict on the test set
test_predictions = clf.predict(test_features)

# Evaluate the classifier
accuracy = accuracy_score(test_labels_encoded, test_predictions)
report = classification_report(test_labels_encoded, test_predictions, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
