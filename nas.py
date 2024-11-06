import os
import argparse
import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time

# Mapping classes to integers (this will be populated based on the classes in the dataset)
class_mapping = {}

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

def main(args):
    # Set parameters
    batch_size = 64
    image_size = (224, 224)
    num_classes = 10
    initial_epochs = 15
    retrain_epochs = 100
    max_trials = 50

    # List class names from the directory and update class_mapping
    class_names = os.listdir(args.data_dir)
    global class_mapping
    class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}

    # Step 1: Load and preprocess the dataset
    data = load_image_data(args.data_dir, image_size=image_size, batch_size=batch_size)
    X, y = dataset_to_numpy(data)
    y_one_hot = to_categorical(y, num_classes=num_classes)

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=121)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=121)

    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    # Step 2: Perform architecture search with AutoKeras
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

    # Step 3: Save the best architecture and retrain
    best_model = clf.export_model()
    best_model.save(args.model_save_path)

    print("Retraining best architecture with more epochs...")
    best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.best_weights_file, monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)

    start_time = time.time()
    best_model.fit(train_data, epochs=retrain_epochs, validation_data=val_data, callbacks=[checkpoint], verbose=1)
    retrain_time = time.time() - start_time
    print(f"Retraining completed in {format_time(retrain_time)}")

    best_model.load_weights(args.best_weights_file)

    # Step 4: Evaluate the model on the test set
    print("Evaluating on the test set...")
    start_time = time.time()
    test_loss, test_accuracy = best_model.evaluate(test_data)
    inference_time = time.time() - start_time
    print(f"Inference time: {format_time(inference_time)}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Step 5: Generate classification metrics and save results
    y_pred = best_model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.result_dir, 'confusion_matrix.png'))
    plt.show()

    # Save classification report
    class_report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    with open(os.path.join(args.result_dir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)
    print("Classification report saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AutoKeras architecture search, retraining, and evaluation on a test set.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the best model")
    parser.add_argument("--best_weights_file", type=str, required=True, help="Path to save the best weights file")

    args = parser.parse_args()
    main(args)
