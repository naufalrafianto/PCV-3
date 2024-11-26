import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.model_selection import train_test_split


class CardClassifier:
    def __init__(self, dataset_path="dataset", img_size=(120, 120)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.class_names = []
        self.model = None

    def load_dataset(self):
        """Load dan preprocess dataset"""
        images = []
        labels = []

        # Load semua kelas
        self.class_names = sorted([d for d in os.listdir(self.dataset_path)
                                   if os.path.isdir(os.path.join(self.dataset_path, d))])
        print(f"Found {len(self.class_names)} classes")

        for idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.dataset_path, class_name)
            print(f"Loading {class_name}...")

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    img = np.expand_dims(img, axis=-1)
                    images.append(img)
                    labels.append(idx)

        X = np.array(images)
        y = keras.utils.to_categorical(labels, len(self.class_names))

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self):
        """Buat arsitektur CNN sederhana"""
        self.model = keras.Sequential([
            # Layer konvolusi pertama
            layers.Conv2D(32, 3, activation='relu', input_shape=(
                self.img_size[0], self.img_size[1], 1)),
            layers.MaxPooling2D(2),

            # Layer konvolusi kedua
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),

            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        """Train model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # Evaluate
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f'\nTest accuracy: {test_acc:.4f}')

        return history

    def save_model(self, filepath="card_classifier_simple.h5"):
        """Simpan model"""
        self.model.save(filepath)

        # Simpan nama kelas
        class_names_file = os.path.splitext(filepath)[0] + "_classes.txt"
        with open(class_names_file, 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")

    def load_model(self, filepath="card_classifier_simple.h5"):
        """Load model"""
        self.model = keras.models.load_model(filepath)

        class_names_file = os.path.splitext(filepath)[0] + "_classes.txt"
        with open(class_names_file, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def predict(self, image):
        """Prediksi satu gambar"""
        if image.shape != self.img_size or len(image.shape) != 2:
            image = cv2.resize(image, self.img_size)

        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)

        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        return self.class_names[predicted_class], confidence


def main():
    # Inisialisasi classifier
    classifier = CardClassifier(img_size=(120, 120))

    # Load dataset
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = classifier.load_dataset()
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Build dan train model
    print("\nBuilding and training model...")
    classifier.build_model()
    classifier.train(X_train, y_train, X_test, y_test, epochs=10)

    # Simpan model
    classifier.save_model()


if __name__ == "__main__":
    main()
