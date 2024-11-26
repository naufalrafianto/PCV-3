import cv2
import numpy as np
import random
from card_classifier_simple import CardClassifier
from utils.card_detection import detect_card
from utils.image_processing import get_warped_card
import time


class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

        # Mapping nilai kartu
        self.value_map = {
            'ace': 11, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
            'nine': 9, 'ten': 10, 'jack': 10, 'queen': 10,
            'king': 10
        }

    def get_value(self):
        return self.value_map[self.value]

    def __str__(self):
        return f"{self.value} of {self.suit}"


class Game41:
    def __init__(self):
        self.classifier = CardClassifier()
        self.classifier.load_model()

        self.player_cards = []
        self.computer_cards = []
        self.top_card = None
        self.discarded_cards = []

        # Generate deck untuk komputer
        self.deck = self.generate_deck()

        # Status game
        self.player_turn = True
        self.game_over = False

    def generate_deck(self):
        """Generate deck lengkap kecuali kartu player"""
        values = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven',
                  'eight', 'nine', 'ten', 'jack', 'queen', 'king']
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        deck = []

        for suit in suits:
            for value in values:
                deck.append(Card(value, suit))

        random.shuffle(deck)
        return deck

    def scan_player_cards(self):
        """Scan kartu player menggunakan webcam"""
        cap = cv2.VideoCapture(2)
        scanned_cards = []

        print("Scanning player cards...")
        print("Show each card to the camera.")
        print("Press 'c' to capture card, 'q' when done (need 4 cards)")

        while len(scanned_cards) < 4:
            ret, frame = cap.read()
            if not ret:
                continue

            card_found, corners, _, _ = detect_card(frame)

            if card_found and corners is not None:
                _, binary_warped = get_warped_card(frame, corners)
                card_class, confidence = self.classifier.predict(binary_warped)

                if confidence > 0.7:  # Only show high confidence predictions
                    value, suit = card_class.split('_')
                    text = f"Detected: {value} of {suit} ({confidence:.2%})"
                    cv2.putText(frame, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Cards scanned: {len(scanned_cards)}/4", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Scan Cards', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and card_found:
                value, suit = card_class.split('_')
                card = Card(value, suit)
                scanned_cards.append(card)
                print(f"Captured: {card}")
                # Remove card from deck
                self.deck = [c for c in self.deck
                             if not (c.value == value and c.suit == suit)]
                time.sleep(1)  # Delay to prevent multiple captures

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return scanned_cards

    def calculate_score(self, cards):
        """Hitung total score untuk set kartu"""
        # Group kartu berdasarkan suit
        suits = {}
        for card in cards:
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)

        # Hitung score tertinggi dari semua suit
        max_score = 0
        for suit_cards in suits.values():
            score = sum(card.get_value() for card in suit_cards)
            max_score = max(max_score, score)

        return max_score

    def computer_decide_action(self):
        """Logika komputer untuk mengambil keputusan"""
        current_score = self.calculate_score(self.computer_cards)

        # Jika top card bisa meningkatkan score
        if self.top_card:
            temp_cards = self.computer_cards + [self.top_card]
            new_score = self.calculate_score(temp_cards)

            if new_score > current_score:
                return 'take'

        return 'draw'

    def computer_discard(self):
        """Logika komputer untuk membuang kartu"""
        # Cari kartu yang memberikan score tertinggi
        best_score = self.calculate_score(self.computer_cards)
        worst_card = self.computer_cards[0]

        for i, card in enumerate(self.computer_cards):
            temp_cards = self.computer_cards.copy()
            temp_cards.pop(i)
            score = self.calculate_score(temp_cards)

            if score >= best_score:
                best_score = score
                worst_card = card

        self.computer_cards.remove(worst_card)
        return worst_card

    def play(self):
        """Main game loop"""
        # Scan kartu player
        print("\nLet's start the game!")
        self.player_cards = self.scan_player_cards()

        if len(self.player_cards) != 4:
            print("Need exactly 4 cards to start. Game cancelled.")
            return

        # Deal kartu komputer
        self.computer_cards = [self.deck.pop() for _ in range(4)]

        # Main game loop
        while not self.game_over:
            # Tampilkan status game
            print("\n" + "="*50)
            print("Your cards:", ", ".join(str(c) for c in self.player_cards))
            print(f"Your score: {self.calculate_score(self.player_cards)}")
            print(f"Computer cards: {len(self.computer_cards)} cards")
            if self.top_card:
                print(f"Top card: {self.top_card}")

            if self.player_turn:
                # Player's turn
                print("\nYour turn!")
                print("1: Take top card")
                print("2: Draw new card")
                print("3: Knock (End game)")
                choice = input("Your choice (1-3): ")

                if choice == '1' and self.top_card:
                    self.player_cards.append(self.top_card)
                    self.top_card = None
                elif choice == '2' and self.deck:
                    self.player_cards.append(self.deck.pop())
                elif choice == '3':
                    self.game_over = True
                    break
                else:
                    print("Invalid choice!")
                    continue

                # Discard
                print("\nYour cards:", ", ".join(str(c)
                      for c in self.player_cards))
                discard_idx = int(input("Choose card to discard (1-5): ")) - 1
                if 0 <= discard_idx < len(self.player_cards):
                    self.top_card = self.player_cards.pop(discard_idx)

            else:
                # Computer's turn
                print("\nComputer's turn...")
                time.sleep(1)

                action = self.computer_decide_action()
                if action == 'take' and self.top_card:
                    print("Computer takes the top card")
                    self.computer_cards.append(self.top_card)
                    self.top_card = None
                elif self.deck:
                    print("Computer draws a new card")
                    self.computer_cards.append(self.deck.pop())

                # Computer discards
                self.top_card = self.computer_discard()
                print(f"Computer discards: {self.top_card}")

                time.sleep(1)

            # Switch turns
            self.player_turn = not self.player_turn

            # Check if deck is empty
            if not self.deck:
                print("\nDeck is empty! Game ending...")
                self.game_over = True

        # Game over - calculate final scores
        player_score = self.calculate_score(self.player_cards)
        computer_score = self.calculate_score(self.computer_cards)

        print("\n" + "="*50)
        print("Game Over!")
        print(f"Your final cards: {', '.join(str(c)
              for c in self.player_cards)}")
        print(f"Your final score: {player_score}")
        print(f"Computer's final cards: {', '.join(
            str(c) for c in self.computer_cards)}")
        print(f"Computer's final score: {computer_score}")

        if player_score > computer_score:
            print("You win!")
        elif computer_score > player_score:
            print("Computer wins!")
        else:
            print("It's a tie!")


def main():
    game = Game41()
    game.play()


if __name__ == "__main__":
    main()
