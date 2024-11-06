import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(test_dir, weights_path):
    # Load the model weights
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape)
    data_augmentation = tf.keras.Sequential(
        [tf.keras.layers.RandomRotation((0.1)), tf.keras.layers.RandomFlip('vertical')])
    augmented_inputs = data_augmentation(inputs)
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=augmented_inputs)
    base_model.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights(weights_path)

    # Set up ImageDataGenerator for the test set
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )

    # Get true labels and predictions
    y_true = test_generator.classes  # True labels from the directory
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(test_generator.class_indices))
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    logloss = log_loss(y_true_one_hot, y_pred_probs)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Log Loss: {logloss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a new test dataset")
    parser.add_argument("--test_dir", type=str, default='ai_test_data', help="Path to the test dataset directory")
    parser.add_argument("--weights_path", type=str, default='model/best_model.weights.h5', help="Path to the trained model weights file")

    args = parser.parse_args()

    evaluate_model(args.test_dir, args.weights_path)
