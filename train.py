import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, log_loss
import matplotlib.pyplot as plt

def main(args):
    # Parameters
    batch_size = 16
    image_size = (224, 224)
    num_classes = 10
    epochs = 50

    # Load the dataset
    train_data = image_dataset_from_directory(
        args.data_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=121,
        shuffle=True
    )
    val_data = image_dataset_from_directory(
        args.data_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=121,
        shuffle=False
    )

    # Build the model
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomFlip('vertical')
    ])
    augmented_inputs = data_augmentation(inputs)
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=augmented_inputs)
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Model Checkpoint
    checkpoint = ModelCheckpoint(args.best_weights_file, monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)

    # Train the model
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[checkpoint])

    # Plot and save training and validation loss and accuracy curves
    plt.figure(figsize=(4,3))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.result_dir, 'training_validation_loss_curve.jpg'), dpi=600, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(4,3))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.result_dir, 'training_validation_accuracy_curve.jpg'), dpi=600, bbox_inches='tight')
    plt.show()

    # Load best weights for evaluation
    model.load_weights(args.best_weights_file)

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_data)
    y_val_true = np.concatenate([y for x, y in val_data], axis=0)
    y_val_pred_probs = model.predict(val_data)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    val_log_loss = log_loss(y_val_true, y_val_pred_probs)
    val_precision = precision_score(y_val_true, y_val_pred, average='weighted')
    val_recall = recall_score(y_val_true, y_val_pred, average='weighted')
    val_f1 = f1_score(y_val_true, y_val_pred, average='weighted')

    # Print validation metrics
    print(f"Validation - Loss: {val_loss}, Accuracy: {val_accuracy}, Log Loss: {val_log_loss}, Precision: {val_precision}, Recall: {val_recall}, F1-Score: {val_f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an Xception model on a dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the training dataset directory")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to save results like plots and model weights")
    parser.add_argument("--best_weights_file", type=str, required=True, help="Path to save the best weights file")
    
    args = parser.parse_args()
    main(args)
