import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import zipfile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import cProfile
import pstats

from sklearn import svm
from sklearn.metrics import accuracy_score

print("Start")

IMAGE_SIZE = 224 
def augment_image(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.5, 1.5))
    return np.array(image)

def open_images(paths, image_size):
    images = []
    for path in paths:
        image = load_img(path, target_size=(image_size, image_size))
        image = augment_image(image)
        images.append(image)
    return np.array(images)

def extract_features(dataset_dir, model, max_steps=100):
    features = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    label_dict = {class_name: idx for idx, class_name in enumerate(class_names)}

    step_count = 0  

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))  # Use IMAGE_SIZE here
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            feature = model.predict(img_array)
            features.append(feature.flatten()) 
            labels.append(label_dict[class_name])

            step_count += 1
            if step_count >= max_steps:
                break 

        if step_count >= max_steps:
            break 

    return np.array(features), np.array(labels)

def flatten_features(features):
    return features.reshape(features.shape[0], -1)

def train_svm_classifier(features, labels, split_ratio=0.8):
    unique_classes = np.unique(labels)
    
    if len(unique_classes) < 2:
        print("Error: The dataset should contain at least two classes.")
        return None
    
    combined_data = np.column_stack((features, labels))
    np.random.shuffle(combined_data)

    features = combined_data[:, :-1]
    labels = combined_data[:, -1]

    split_idx = int(len(features) * split_ratio)
    train_features, test_features = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(train_features, train_labels)
    
    return clf, test_features, test_labels

def main():
   
    train_paths = [...]  # Your list of training paths
    train_labels = [...]  # Your list of training labels

    images = open_images(train_paths[50:59], IMAGE_SIZE)
    labels = train_labels[50:59]
    
    # Example usage:
    train_dir = '/content/melanoma_cancer_dataset/train'
    test_dir = '/content/melanoma_cancer_dataset/test'

    model = VGG16(weights='imagenet', input_shape=(224, 224, 3))
    model = Model(inputs=model.input, outputs=model.get_layer("fc1").output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    features, labels = extract_features(train_dir, model, max_steps=100)
    features = flatten_features(features)

    result = train_svm_classifier(features, labels)
    
    if result is not None:
        clf, test_features, test_labels = result

        if clf is not None:
            # Testing SVM classifier and getting accuracy
            predictions = clf.predict(test_features)
            accuracy = accuracy_score(test_labels, predictions)
            print(f"Accuracy: {accuracy * 100:.2f}%")
        
    # Stop profiling
    profiler.disable()
    profiler.dump_stats("profile_results.prof")

    # Load and print the profile results
    with open("profile_results.prof", "rb") as f:
        stats = pstats.Stats(f)
        stats.strip_dirs()
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()

if __name__ == "__main__":
    main()
