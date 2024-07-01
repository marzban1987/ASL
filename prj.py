import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

# Paths to train and test directories
train_dir = 'Train'
test_dir = 'Test'

# Function to load images and labels for all letters in a directory
def load_images_for_all_letters(folder):
    images = []
    labels = []
    for letter in os.listdir(folder):
        letter_folder = os.path.join(folder, letter)
        if os.path.isdir(letter_folder):
            for file in os.listdir(letter_folder):
                img_path = os.path.join(letter_folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(letter)
    return images, labels

# Load training and testing data for all letters
train_images, train_labels = load_images_for_all_letters(train_dir)
test_images, test_labels = load_images_for_all_letters(test_dir)

# Initialize ORB detector
orb = cv2.ORB_create()

# Function to extract ORB features
def extract_orb_features(images):
    descriptors_list = []
    valid_images = []
    for img in images:
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
            valid_images.append(img)
    return descriptors_list, valid_images

# Extract ORB features from training and testing images
train_descriptors, valid_train_images = extract_orb_features(train_images)
test_descriptors, valid_test_images = extract_orb_features(test_images)

# Flatten the list of descriptors
all_descriptors = np.vstack(train_descriptors)

# Define the number of clusters
num_clusters = 100

# Perform k-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_descriptors)

# Function to convert descriptors to histogram of visual words
def descriptors_to_hist(descriptors_list, kmeans):
    hist_list = []
    for descriptors in descriptors_list:
        words, _ = vq(descriptors, kmeans.cluster_centers_)
        hist, _ = np.histogram(words, bins=np.arange(num_clusters + 1), density=True)
        hist_list.append(hist)
    return hist_list

# Convert descriptors to histograms
train_features = descriptors_to_hist(train_descriptors, kmeans)
test_features = descriptors_to_hist(test_descriptors, kmeans)

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
