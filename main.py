import cv2
import numpy as np
import random
import os
import time
from card_classifier_simple import CardClassifier
from utils.card_detection import detect_card
from utils.image_processing import get_warped_card

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
        
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

class CardDisplay:
    def __init__(self, cards_dir="./card-image"):
        self.cards_dir = cards_dir
        self.card_images = {}
        self.card_back = None
        self.background = None
        self.CARD_WIDTH = 120   # Ukuran lebih kecil
        self.CARD_HEIGHT = 180 
        self.load_card_images()
        
    def load_card_images(self):
        if not os.path.exists(self.cards_dir):
            print(f"Warning: Directory {self.cards_dir} not found!")
            return
            
        # Load card back image
        back_path = os.path.join(self.cards_dir, "flip-card.png")
        if os.path.exists(back_path):
            self.card_back = cv2.imread(back_path)
            self.card_back = cv2.resize(self.card_back, (self.CARD_WIDTH, self.CARD_HEIGHT))
        else:
            print("Warning: flip-card.png not found!")
            
        # Load background image
        bg_path = os.path.join(self.cards_dir, "card_back.png")
        if os.path.exists(bg_path):
            self.background = cv2.imread(bg_path)
        else:
            print("Warning: card_back.png not found!")
            
        # Load card images
        for filename in os.listdir(self.cards_dir):
            if filename.endswith('.png') and filename not in ['card_back.png', 'flip-card.png']:
                card_name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.cards_dir, filename)
                card_img = cv2.imread(img_path)
                if card_img is not None:
                    card_img = cv2.resize(card_img, (self.CARD_WIDTH, self.CARD_HEIGHT))
                    self.card_images[card_name] = card_img
        
        print(f"Loaded {len(self.card_images)} card images")
    
    def get_card_image(self, value, suit):
        value_map = {
            'ace': 'A', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
            'nine': '9', 'ten': '10', 'jack': 'J', 'queen': 'Q',
            'king': 'K'
        }
        filename = f"{suit}_{value_map[value]}"
        return self.card_images.get(filename)

class Game41Board:
    def __init__(self):
        self.classifier = CardClassifier()
        self.classifier.load_model()
        self.card_display = CardDisplay()

        self.CARD_WIDTH = self.card_display.CARD_WIDTH
        self.CARD_HEIGHT = self.card_display.CARD_HEIGHT
        self.CARD_MARGIN = 10  # Margin antar kartu lebih kecil
        
        # Update posisi area kartu
        self.COMPUTER_AREA = (50, 30)   # Sesuaikan posisi area computer
        self.PLAYER_AREA = (50, 500)    # Sesuaikan posisi area player
        self.DECK_AREA = (400, 250)     # Sesuaikan posisi deck
        self.TOP_CARD_AREA = (550, 250) 
        
        # Window setup
        cv2.namedWindow('Game 41', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Game 41', 1280, 720)
        
        # Game state
        self.player_cards = []
        self.computer_cards = []
        self.top_card = None
        self.deck = self.generate_deck()
        self.game_over = False
        self.player_turn = True
        self.selected_card = None 
        
        # Display settings
        self.board_width = 1280
        self.board_height = 720
        self.card_width = 200
        self.card_height = 300
        
        # Game message
        self.message = ""
        self.message_time = 0

        
        
    def generate_deck(self):
        values = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven',
                 'eight', 'nine', 'ten', 'jack', 'queen', 'king']
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        deck = []
        
        for suit in suits:
            for value in values:
                deck.append(Card(value, suit))
        
        random.shuffle(deck)
        return deck
    
    def overlay_image(self, background, overlay, position):
        x, y = position
        h, w = overlay.shape[:2]
        
        if y + h > background.shape[0] or x + w > background.shape[1]:
            return
            
        roi = background[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(overlay, overlay, mask=mask)
        
        background[y:y+h, x:x+w] = cv2.add(bg, fg)
    
    def draw_card(self, image, card, position, face_up=True):
        if face_up and card:
            card_img = self.card_display.get_card_image(card.value, card.suit)
            if card_img is not None:
                self.overlay_image(image, card_img, position)
        else:
            if self.card_display.card_back is not None:
                self.overlay_image(image, self.card_display.card_back, position)
    
    def draw_text_with_background(self, image, text, position, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        x, y = position
        cv2.rectangle(image, (x-5, y-text_height-5), (x+text_width+5, y+5), 
                     (0, 0, 0), -1)
        cv2.putText(image, text, position, font, scale, color, thickness)
    
    def draw_board(self):
        if self.card_display.background is not None:
            board = cv2.resize(self.card_display.background, (self.board_width, self.board_height))
        else:
            board = np.ones((self.board_height, self.board_width, 3), dtype=np.uint8) * 50
        
        # Draw computer cards
        for i, card in enumerate(self.computer_cards):
            pos = (self.COMPUTER_AREA[0] + i * (self.CARD_WIDTH + self.CARD_MARGIN),
                  self.COMPUTER_AREA[1])
            self.draw_card(board, card, pos, face_up=False)
        
        # Draw deck and top card
        if self.deck:
            self.draw_card(board, None, (500, 250), face_up=False)
        if self.top_card:
            self.draw_card(board, self.top_card, (750, 250), face_up=True)
        
        # Draw player cards
        for i, card in enumerate(self.player_cards):
            pos = (self.PLAYER_AREA[0] + i * (self.CARD_WIDTH + self.CARD_MARGIN),
                  self.PLAYER_AREA[1])
            self.draw_card(board, card, pos, face_up=True)

            if i == self.selected_card:
                cv2.rectangle(board,
                            (pos[0]-2, pos[1]-2),
                            (pos[0]+self.CARD_WIDTH+2, pos[1]+self.CARD_HEIGHT+2),
                            self.COLORS['highlight'], 2)
        
        # Draw scores
        player_score = self.calculate_score(self.player_cards)
        computer_score = self.calculate_score(self.computer_cards)
        
        self.draw_text_with_background(board, f"Your Score: {player_score}", (900, 450))
        self.draw_text_with_background(board, f"Computer Score: {computer_score}", (900, 50))
        
        # Draw turn indicator
        turn_text = "Your Turn" if self.player_turn else "Computer's Turn"
        self.draw_text_with_background(board, turn_text, (900, 250), 
                                     color=(0, 255, 0) if self.player_turn else (0, 0, 255))
        
        # Draw message if exists
        if self.message and time.time() - self.message_time < 3:
            self.draw_text_with_background(board, self.message, (400, 350), color=(255, 255, 0))
        
        # Draw controls help
        controls = "Controls: [A] Take top card  [S] Draw new card  [D] Knock  [Q] Quit"
        self.draw_text_with_background(board, controls, (50, 680), color=(200, 200, 200))
        
        return board
    
    def calculate_score(self, cards):
        suits = {}
        for card in cards:
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)
        
        max_score = 0
        for suit_cards in suits.values():
            score = sum(card.get_value() for card in suit_cards)
            max_score = max(max_score, score)
        
        return max_score
    
    def scan_cards(self):
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
                
                if confidence > 0.7:
                    value, suit = card_class.split('_')
                    text = f"Detected: {value} of {suit} ({confidence:.2%})"
                    cv2.putText(frame, text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show detected card image
                    card_img = self.card_display.get_card_image(value, suit)
                    if card_img is not None:
                        cv2.imshow('Detected Card', card_img)
            
            cv2.putText(frame, f"Cards scanned: {len(scanned_cards)}/4",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Scan Cards', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and card_found:
                value, suit = card_class.split('_')
                card = Card(value, suit)
                scanned_cards.append(card)
                print(f"Captured: {card}")
                
                # Remove from deck
                self.deck = [c for c in self.deck
                           if not (c.value == value and c.suit == suit)]
                time.sleep(1)
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyWindow('Scan Cards')
        cv2.destroyWindow('Detected Card')
        
        return scanned_cards
    
    def computer_turn(self):
        # Tampilkan animasi "thinking" sebelum mengambil keputusan
        thinking_duration = 1.5  # Durasi berpikir dalam detik
        start_time = time.time()
        dots = 0
        
        while time.time() - start_time < thinking_duration:
            # Update board dengan animasi
            board = self.draw_board()  # Gambar board normal
            
            # Animasi titik-titik
            dots = (dots + 1) % 4
            thinking_text = "Computer thinking" + "." * dots
            
            # Tambahkan text thinking
            self.draw_text_with_background(
                board, 
                thinking_text,
                (400, 350),  # Posisi text di tengah board
                color=(255, 255, 0)  # Warna kuning
            )
            
            cv2.imshow('Game 41', board)
            cv2.waitKey(250)  # Update setiap 250ms
        
        # Simple AI logic setelah animasi
        current_score = self.calculate_score(self.computer_cards)
        
        # Try to take top card if it improves score
        if self.top_card:
            temp_cards = self.computer_cards + [self.top_card]
            new_score = self.calculate_score(temp_cards)
            
            if new_score > current_score:
                self.computer_cards.append(self.top_card)
                self.top_card = None
                self.message = "Computer takes top card"
            # If didn't take top card, draw from deck
            elif self.deck:
                self.computer_cards.append(self.deck.pop())
                self.message = "Computer draws new card"
            
            # Always discard a card if we have more than 4
            if len(self.computer_cards) > 4:
                worst_card = min(self.computer_cards, key=lambda c: c.get_value())
                self.computer_cards.remove(worst_card)
                self.top_card = worst_card
                self.message = f"Computer discards {worst_card}"
    
    def play(self):
        # Initial setup
        self.player_cards = self.scan_cards()
        if len(self.player_cards) != 4:
            return
        
        # Give computer exactly 4 cards
        self.computer_cards = [self.deck.pop() for _ in range(4)]
        
        while not self.game_over:
            # Check scores
            player_score = self.calculate_score(self.player_cards)
            computer_score = self.calculate_score(self.computer_cards)
            
            if player_score >= 41 or computer_score >= 41:
                self.game_over = True
                break
            
            # Draw and show board
            board = self.draw_board()
            cv2.imshow('Game 41', board)
            
            if self.player_turn:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('a') and self.top_card:  # Take top card
                    self.player_cards.append(self.top_card)
                    self.top_card = None
                    self.message = "Select a card to discard (1-5)"
                    self.message_time = time.time()
                    
                elif key == ord('s') and self.deck:  # Draw new card
                    self.player_cards.append(self.deck.pop())
                    self.message = "Select a card to discard (1-5)"
                    self.message_time = time.time()
                    
                elif key == ord('d'):  # Knock
                    self.game_over = True
                    break
                    
                elif key in [ord(str(i)) for i in range(1, 6)] and len(self.player_cards) > 4:
                    # Discard selected card
                    idx = int(chr(key)) - 1
                    if idx < len(self.player_cards):
                        self.top_card = self.player_cards.pop(idx)
                        self.player_turn = False
                        self.message = ""
                
                elif key == ord('q'):
                    break
                    
            else:
                # Computer's turn
                time.sleep(1)
                self.computer_turn()
                self.message_time = time.time()
                self.player_turn = True
            
            if not self.deck:
                self.game_over = True
        
        # Show final scores
        player_score = self.calculate_score(self.player_cards)
        computer_score = self.calculate_score(self.computer_cards)
        # Show final scores and determine winner
        final_board = self.draw_board()
        
        # Draw final message
        winner_msg = ""
        if player_score >= 41:
            winner_msg = "You win with 41 points!"
        elif computer_score >= 41:
            winner_msg = "Computer wins with 41 points!"
        elif player_score > computer_score:
            winner_msg = f"You win! {player_score} vs {computer_score}"
        elif computer_score > player_score:
            winner_msg = f"Computer wins! {computer_score} vs {player_score}"
        else:
            winner_msg = f"It's a tie! {player_score} points"

        # Draw winner message
        self.draw_text_with_background(final_board, winner_msg, 
                                     (self.board_width//2 - 200, self.board_height//2),
                                     color=(0, 255, 255))
        
        # Show final state
        cv2.imshow('Game 41', final_board)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()

def main():
    while True:
        # Create game instance
        game = Game41Board()
        
        # Show welcome screen
        welcome_board = np.ones((game.board_height, game.board_width, 3), dtype=np.uint8) * 50
        game.draw_text_with_background(welcome_board, "Welcome to Game 41!", 
                                     (game.board_width//2 - 150, game.board_height//2 - 100))
        game.draw_text_with_background(welcome_board, "Press SPACE to start or Q to quit", 
                                     (game.board_width//2 - 200, game.board_height//2))
        cv2.imshow('Game 41', welcome_board)
        
        # Wait for player input
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to start
                break
            elif key == ord('q'):  # Q to quit
                cv2.destroyAllWindows()
                return
        
        # Start game
        game.play()
        
        # Ask for replay
        replay_board = np.ones((game.board_height, game.board_width, 3), dtype=np.uint8) * 50
        game.draw_text_with_background(replay_board, "Play again? (Y/N)", 
                                     (game.board_width//2 - 100, game.board_height//2))
        cv2.imshow('Game 41', replay_board)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                break
            elif key == ord('n') or key == ord('q'):
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()