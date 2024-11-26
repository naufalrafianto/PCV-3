import os
import cv2
import time
from utils.card_detection import detect_card
from utils.image_processing import get_warped_card


class CardDatasetGenerator:
    def __init__(self, output_dir="dataset"):
        self.output_dir = output_dir
        self.card_names = {
            'A': 'ace',   '2': 'two',   '3': 'three',
            '4': 'four',  '5': 'five',  '6': 'six',
            '7': 'seven', '8': 'eight', '9': 'nine',
            '0': 'ten',   'J': 'jack',  'Q': 'queen',
            'K': 'king'
        }
        self.suits = {
            'H': 'hearts',
            'D': 'diamonds',
            'C': 'clubs',
            'S': 'spades'
        }
        self.current_card = None
        self.current_count = 0
        self.max_samples = 50
        self.last_save_time = 0
        self.save_delay = 0.1  # 100ms delay between saves

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def set_current_card(self, shortcut):
        """Set kartu menggunakan shortcut (contoh: 'QH' untuk Queen of Hearts)"""
        if len(shortcut) < 2:
            return False

        value = shortcut[0].upper()
        suit = shortcut[1].upper()

        # Handle angka 10
        if value == '1' and len(shortcut) >= 3 and shortcut[1] == '0':
            value = '0'  # menggunakan '0' untuk merepresentasikan 10
            suit = shortcut[2].upper()

        if value in self.card_names and suit in self.suits:
            self.current_card = f"{self.card_names[value]}_{self.suits[suit]}"
            self.current_count = 0

            card_dir = os.path.join(self.output_dir, self.current_card)
            if not os.path.exists(card_dir):
                os.makedirs(card_dir)

            return True
        return False

    def save_card_image(self, binary_image):
        """Simpan gambar kartu ke dataset"""
        current_time = time.time()
        if current_time - self.last_save_time < self.save_delay:
            return -1

        if self.current_card and self.current_count < self.max_samples:
            card_dir = os.path.join(self.output_dir, self.current_card)
            filename = f"{self.current_card}_{self.current_count:03d}.png"
            filepath = os.path.join(card_dir, filename)

            cv2.imwrite(filepath, binary_image)
            self.current_count += 1
            self.last_save_time = current_time

            return self.current_count
        return -1


def print_help():
    print("\nPanduan Shortcut:")
    print("1. Input format: [Nilai][Jenis]")
    print("   Nilai: A,2-9,0(untuk 10),J,Q,K")
    print("   Jenis: H(Hearts), D(Diamonds), C(Clubs), S(Spades)")
    print("   Contoh: QH = Queen of Hearts, 0D = Ten of Diamonds")
    print("\n2. Kontrol:")
    print("   SPACE = Auto generate 50 gambar")
    print("   H = Tampilkan panduan ini")
    print("   Q = Keluar")
    print("\nContoh input lengkap:")
    print("AH = Ace of Hearts    | 0H = Ten of Hearts")
    print("2D = Two of Diamonds  | JD = Jack of Diamonds")
    print("3C = Three of Clubs   | QC = Queen of Clubs")
    print("4S = Four of Spades   | KS = King of Spades")


def generate_dataset():
    cap = cv2.VideoCapture(2)
    generator = CardDatasetGenerator()

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Binary Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', 600, 400)
    cv2.resizeWindow('Binary Result', 400, 600)

    print_help()
    current_input = ""
    is_capturing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        card_found, corners, _, _ = detect_card(frame)

        # Tampilkan frame original
        cv2.putText(frame, f"Input: {current_input}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if generator.current_card:
            cv2.putText(frame, f"Current: {generator.current_card}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {generator.current_count}/{generator.max_samples}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Original', frame)

        if card_found and corners is not None:
            _, binary_warped = get_warped_card(frame, corners)
            cv2.imshow('Binary Result', binary_warped)

            # Auto capture jika sedang dalam mode capturing
            if is_capturing and generator.current_card:
                count = generator.save_card_image(binary_warped)
                if count > 0:
                    print(f"Saved: {generator.current_card}_{count:03d}.png")
                    if count >= generator.max_samples:
                        print(f"\nSelesai mengumpulkan {
                              generator.max_samples} gambar untuk {generator.current_card}")
                        is_capturing = False
                        generator.current_card = None
                        current_input = ""

        key = cv2.waitKey(1) & 0xFF

        if key == ord('/'):
            break
        elif key == ord(';'):
            print_help()
        elif key == 32:  # SPACE
            if generator.current_card and card_found:
                is_capturing = True
                print(f"\nMulai auto-capture untuk {generator.current_card}")
        elif key in [ord(str(i)) for i in range(10)] or \
                key in [ord(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.lower()]:
            current_input += chr(key).upper()
            if len(current_input) >= 2:
                if generator.set_current_card(current_input):
                    print(f"\nKartu diset ke: {generator.current_card}")
                    print("Tekan SPACE untuk mulai auto-generate 50 gambar")
                else:
                    print(f"Input tidak valid: {current_input}")
                    print("Tekan H untuk panduan format input")
                current_input = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_dataset()
