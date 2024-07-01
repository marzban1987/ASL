import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

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

# Load training data for letters 'A' and 'B'
train_images, train_labels = load_images_for_letters(train_dir, ['A', 'B'])

# Initialize ORB detector
orb = cv2.ORB_create()

# Function to extract ORB features
def extract_orb_features(images):
    keypoints_list = []
    descriptors_list = []
    for img in images:
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is not None:
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
    return keypoints_list, descriptors_list

# Extract ORB features from training images
train_keypoints, train_descriptors = extract_orb_features(train_images)

# Plot the ORB features
def plot_orb_features(images, keypoints_list):
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        img = images[i]
        keypoints = keypoints_list[i]
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.title(f'Image {i + 1}')
    plt.show()

# Plot ORB features for the first few images
plot_orb_features(train_images[:5], train_keypoints[:5])







