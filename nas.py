import os
import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import time

# Set your data paths and result directory
data_dir = '/data/home/mk1723/CampusVisionChallengeFinal/MergeDataset'
result_dir = '/data/home/mk1723/soybean/results'

# Parameters
batch_size = 64
image_size = (224, 224)
num_classes = 10
initial_epochs = 15  # For architecture search
retrain_epochs = 100  # For retraining the best architecture
max_trials = 50  # AutoKeras search space trials

# List class names from the directory
class_names = os.listdir(data_dir)

# Mapping classes to integers
class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

# Function to format time
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# Function to load and preprocess image data using AutoKeras's image_dataset_from_directory
def load_image_data(data_dir, image_size=(224, 224), batch_size=32):
    data = ak.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=121,
        shuffle=True
    )
    return data

# Convert datasets to NumPy arrays for splitting and one-hot encode labels
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image, label in dataset.unbatch():
        images.append(image.numpy())
        labels.append(label.numpy())
    labels = np.array([class_mapping[label.decode()] for label in labels])  # Map string labels to integers
    return np.array(images), labels

# Step 1: Load the full dataset
data = load_image_data(data_dir, image_size=image_size, batch_size=batch_size)

# Convert the dataset to NumPy arrays for splitting
X, y = dataset_to_numpy(data)

# Print the total number of images in the dataset
print(f"Total number of images in the dataset: {len(X)}")

# One-hot encode labels for training
y_one_hot = to_categorical(y, num_classes=num_classes)

# Step 2: Split the data once (80% train/validation, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=121)

# Print the number of images in the initial split
print(f"Number of images in the 80% training/validation split: {len(X_train_val)}")
print(f"Number of images in the 20% test split: {len(X_test)}")

# Step 3: Further split the 80% into 80% train and 20% validation for both architecture search and retraining
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=121)

# Print the number of images in the training and validation split for architecture search and retraining
print(f"Number of images used for training: {len(X_train)}")
print(f"Number of images used for validation: {len(X_val)}")

# Create datasets for training, validation, and testing
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# Step 4: Perform AutoKeras architecture search (using train_data and val_data)
input_node = ak.ImageInput()
output_node = ak.ImageBlock(block_type='xception')(input_node)
output_node = ak.ClassificationHead()(output_node)

clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=True,
    max_trials=max_trials,
    objective='val_accuracy',
    project_name='autokeras_search'
)

print("Starting architecture search...")
start_time = time.time()
clf.fit(train_data, epochs=initial_epochs, validation_data=val_data, verbose=1)
arch_search_time = time.time() - start_time
print(f"Architecture search completed in {format_time(arch_search_time)}")

# Step 5: Save the best architecture found during the search
best_model = clf.export_model()
model_save_path = '/data/home/mk1723/CampusVisionChallengeFinal/best_model.keras'  
best_model.save(model_save_path)

# Step 6: Retrain the best architecture using the same split for consistency
print("Retraining best architecture with more epochs...")

# Print the number of images used for retraining (same split as before)
print(f"Number of images used for retraining: {len(X_train)}")

# Compile the best model
best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the best weights during retraining
best_weights_file = '/data/home/mk1723/CampusVisionChallengeFinal/best_weights.weights.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_weights_file, monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)

start_time = time.time()
best_model.fit(train_data, epochs=retrain_epochs, validation_data=val_data, callbacks=[checkpoint], verbose=1)
retrain_time = time.time() - start_time
print(f"Retraining completed in {format_time(retrain_time)}")

# Load the best weights after retraining
best_model.load_weights(best_weights_file)

# Step 7: Evaluate the model on the held-out 20% test set
print("Evaluating on the test set...")
start_time = time.time()
test_loss, test_accuracy = best_model.evaluate(test_data)
inference_time = time.time() - start_time
print(f"Inference time: {format_time(inference_time)}")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Generate confusion matrix and classification report
y_pred = best_model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Step 8: Confusion matrix and saving results
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
confusion_matrix_path = '/data/home/mk1723/CampusVisionChallengeFinal/cm.png'
plt.savefig(confusion_matrix_path)
plt.show()

# Save classification report
class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
classification_path = os.path.join(result_dir, 'classification_report.txt')
with open(classification_path, 'w') as f:
    f.write(class_report)
print("Classification report saved.")
