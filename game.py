import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import os
import sys
from typing import List, Dict, Optional, Tuple

class ShapeMatcherGame:
    """Main class for the gesture-controlled shape matching game."""
    
    def __init__(self):
        """Initialize the game with default settings and resources."""
        self._initialize_constants()
        self._initialize_pygame()
        self._load_assets()
        self._setup_webcam()
        self._setup_hand_detection()
        
    def _initialize_constants(self):
        """Set up game constants and configuration."""
        self.PIECE_SIZE = 100
        self.SNAP_THRESHOLD = 40
        self.SHAPE_NAMES = ["circle", "square", "rectangle", "triangle"]
        self.TARGET_POSITIONS = [(25, 100), (200, 100), (375, 100), (500, 100)]
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 800, 600
        
    def _initialize_pygame(self):
        """Initialize pygame and create game window."""
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("VisionShape Matcher")
        
    def _load_assets(self):
        """Load game assets including images and sounds."""
        self.success_sound = self._load_sound("success.wav")
        self.shape_pieces = self._load_shape_pieces()
        
    def _load_sound(self, filename: str) -> Optional[pygame.mixer.Sound]:
        """Attempt to load a sound file with error handling."""
        if not os.path.exists(filename):
            print(f"⚠️ Sound file not found: {filename}")
            return None
            
        try:
            return pygame.mixer.Sound(filename)
        except Exception as e:
            print(f"⚠️ Could not load sound {filename}: {e}")
            return None
            
    def _load_shape_pieces(self) -> List[Dict]:
        """Load and validate shape images from files."""
        pieces = []
        for i, name in enumerate(self.SHAPE_NAMES):
            try:
                img = self._load_shape_image(name)
                pieces.append({
                    "name": name,
                    "image": img,
                    "position": [100 + i * 120, 300],
                    "target": self.TARGET_POSITIONS[i],
                    "is_placed": False,
                    "is_dragging": False
                })
            except Exception as e:
                print(f"❌ Failed to load shape {name}: {e}")
                sys.exit(1)
        return pieces
        
    def _load_shape_image(self, name: str) -> np.ndarray:
        """Load a single shape image with validation."""
        path = f"shape_{name}.png"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing shape image: {path}")
            
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] < 4:
            raise ValueError(f"{path} must be a PNG with transparency")
            
        return cv2.resize(img, (self.PIECE_SIZE, self.PIECE_SIZE))
        
    def _setup_webcam(self):
        """Initialize and validate webcam access."""
        self.cap = self._find_working_camera()
        if self.cap is None:
            print("❌ Could not access any webcam.")
            print("🔧 Make sure a webcam is connected and not in use.")
            sys.exit(1)
            
        cv2.namedWindow('VisionShape Matcher', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('VisionShape Matcher', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("✅ Webcam accessed successfully.")
        
    def _find_working_camera(self, max_index: int = 4) -> Optional[cv2.VideoCapture]:
        """Try to find an available camera device."""
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                return cap
        return None
        
    def _setup_hand_detection(self):
        """Initialize MediaPipe hand detection."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
    def run(self):
        """Main game loop."""
        print("🎮 Game started! Use a pinch gesture to drag and match the shapes.")
        
        while True:
            frame = self._capture_frame()
            if frame is None:
                break
                
            processed_frame = self._process_frame(frame)
            cv2.imshow("VisionShape Matcher", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
                
        self._cleanup()
        
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Frame not received from webcam.")
            return None
        return cv2.flip(frame, 1)
        
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with hand detection and game logic."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)
        
        pinch_state = self._detect_pinch_gesture(hand_results, frame.shape)
        self._update_shape_positions(pinch_state)
        
        self._draw_targets(frame)
        self._draw_shape_pieces(frame)
        
        if self._check_win_condition():
            self._draw_win_message(frame)
            
        return frame
        
    def _detect_pinch_gesture(self, hand_results, frame_shape) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Detect pinch gesture from hand landmarks."""
        if not hand_results.multi_hand_landmarks:
            return False, None
            
        h, w, _ = frame_shape
        pinch_detected = False
        pinch_position = None
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            thumb = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            index = (int(landmarks[8].x * w), int(landmarks[8].y * h))
            middle = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            
            # Check distance between fingers for pinch
            if (self._calculate_distance(thumb, index) < 40 and 
                self._calculate_distance(thumb, middle) < 40):
                pinch_detected = True
                pinch_position = (
                    (index[0] + middle[0]) // 2,
                    (index[1] + middle[1]) // 2
                )
                
            # Draw hand landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
        return pinch_detected, pinch_position
        
    def _update_shape_positions(self, pinch_state: Tuple[bool, Optional[Tuple[int, int]]):
        """Update shape positions based on pinch gesture."""
        pinch_detected, pinch_position = pinch_state
        
        for piece in self.shape_pieces:
            if piece["is_placed"]:
                continue
                
            if pinch_detected and pinch_position:
                self._handle_dragging(piece, pinch_position)
            elif piece["is_dragging"]:
                self._handle_shape_placement(piece)
                
    def _handle_dragging(self, piece: Dict, pinch_position: Tuple[int, int]):
        """Handle the dragging logic for a shape piece."""
        x, y = pinch_position
        piece_x, piece_y = piece["position"]
        
        # Check if pinch is inside piece or already dragging
        if (self._is_point_in_shape(x, y, piece) or piece["is_dragging"]):
            piece["position"][0] = x - self.PIECE_SIZE // 2
            piece["position"][1] = y - self.PIECE_SIZE // 2
            piece["is_dragging"] = True
            
    def _handle_shape_placement(self, piece: Dict):
        """Handle shape placement and snapping to target."""
        dx = piece["position"][0] - piece["target"][0]
        dy = piece["position"][1] - piece["target"][1]
        
        if abs(dx) < self.SNAP_THRESHOLD and abs(dy) < self.SNAP_THRESHOLD:
            piece["position"] = list(piece["target"])
            piece["is_placed"] = True
            if self.success_sound:
                self.success_sound.play()
                
        piece["is_dragging"] = False
        
    def _draw_targets(self, frame: np.ndarray):
        """Draw target positions for each shape."""
        for i, target in enumerate(self.TARGET_POSITIONS):
            shape = self.SHAPE_NAMES[i]
            x, y = target
            
            if shape == "circle":
                cv2.circle(frame, (x + 50, y + 50), 45, (0, 255, 255), 3)
            elif shape == "square":
                cv2.rectangle(frame, (x, y), (x + 100, y + 100), (255, 255, 0), 3)
            elif shape == "rectangle":
                cv2.rectangle(frame, (x, y + 25), (x + 100, y + 75), (0, 255, 255), 3)
            elif shape == "triangle":
                pts = np.array([[x + 50, y], [x + 100, y + 100], [x, y + 100]], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=3)
                
    def _draw_shape_pieces(self, frame: np.ndarray):
        """Draw all shape pieces on the frame."""
        for piece in self.shape_pieces:
            self._overlay_image(frame, piece["image"], piece["position"])
            
    def _check_win_condition(self) -> bool:
        """Check if all shapes have been placed correctly."""
        return all(p["is_placed"] for p in self.shape_pieces)
        
    def _draw_win_message(self, frame: np.ndarray):
        """Display win message when all shapes are matched."""
        cv2.putText(frame, " All Shapes Matched!", (140, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                    
    def _overlay_image(self, background: np.ndarray, overlay: np.ndarray, position: List[int]):
        """Overlay an image with transparency onto the background."""
        x, y = position
        h, w = overlay.shape[:2]
        
        if y + h > background.shape[0] or x + w > background.shape[1]:
            return
            
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                background[y:y+h, x:x+w, c] * (1 - alpha) + 
                overlay[:, :, c] * alpha
            )
            
    def _calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        
    def _is_point_in_shape(self, x: int, y: int, piece: Dict) -> bool:
        """Check if a point is inside a shape's bounding box."""
        px, py = piece["position"]
        return (px <= x <= px + self.PIECE_SIZE and 
                py <= y <= py + self.PIECE_SIZE)
                
    def _cleanup(self):
        """Release resources and clean up."""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("👋 Game exited.")

if __name__ == "__main__":
    game = ShapeMatcherGame()
    game.run()