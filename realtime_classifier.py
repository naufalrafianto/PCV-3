import cv2
import os
import numpy as np
from card_classifier_simple import CardClassifier
from utils.card_detection import detect_card
from utils.image_processing import get_warped_card


class CardDisplay:
    def __init__(self, cards_dir="./card-image"):
        self.cards_dir = cards_dir
        self.card_images = {}
        self.load_card_images()

    def load_card_images(self):
        """Load semua gambar kartu digital"""
        if not os.path.exists(self.cards_dir):
            print(f"Warning: Directory {self.cards_dir} not found!")
            return

        for filename in os.listdir(self.cards_dir):
            if filename.endswith('.png'):
                # Remove extension
                card_name = os.path.splitext(filename)[0]
                # Load image
                img_path = os.path.join(self.cards_dir, filename)
                card_img = cv2.imread(img_path)
                if card_img is not None:
                    # Resize untuk display
                    card_img = cv2.resize(card_img, (200, 300))
                    self.card_images[card_name] = card_img

        print(f"Loaded {len(self.card_images)} card images")

    def get_card_image(self, card_class):
        """Get digital image untuk kartu yang terdeteksi"""
        # Convert nama kelas ke format nama file
        # Contoh: "ace_of_hearts" -> "hearts_A"
        try:
            value, suit = card_class.split('_', 1)

            # Mapping nilai
            value_map = {
                'ace': 'A', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
                'nine': '9', 'ten': '10', 'jack': 'J', 'queen': 'Q',
                'king': 'K'
            }

            # Buat nama file
            filename = f"{suit}_{value_map[value]}"

            return self.card_images.get(filename)
        except:
            return None


def main():
    # Initialize classifier dan card display
    classifier = CardClassifier()
    classifier.load_model()
    card_display = CardDisplay()

    cap = cv2.VideoCapture(2)

    # Buat windows
    cv2.namedWindow('Card Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Digital Card', cv2.WINDOW_NORMAL)

    # Set ukuran windows
    cv2.resizeWindow('Card Detection', 800, 600)
    cv2.resizeWindow('Digital Card', 200, 300)

    last_prediction = None
    confidence_threshold = 0.7  # Minimum confidence untuk menampilkan prediksi

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi kartu
        card_found, corners, _, _ = detect_card(frame)

        if card_found and corners is not None:
            # Get warped card image
            _, binary_warped = get_warped_card(frame, corners)

            # Predict
            card_class, confidence = classifier.predict(binary_warped)

            # Update prediksi jika confidence tinggi
            if confidence > confidence_threshold:
                last_prediction = card_class

            # Tampilkan hasil prediksi pada frame
            text = f"{card_class} ({confidence:.2%})"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Tampilkan gambar digital kartu
            if last_prediction:
                digital_card = card_display.get_card_image(last_prediction)
                if digital_card is not None:
                    cv2.imshow('Digital Card', digital_card)

        # Tampilkan frame deteksi
        cv2.imshow('Card Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
