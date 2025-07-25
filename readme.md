<div align="center"> <h1>👋 GestureShape Matcher</h1> <p><em>Match shapes with your hands using AI-powered gesture controls!</em></p>
https://demo.gif <!-- Replace with actual demo later -->
https://img.shields.io/badge/OpenCV-5.0+-orange
https://img.shields.io/badge/MediaPipe-0.9.1-blue
https://img.shields.io/badge/PyGame-2.1.3-yellow
https://img.shields.io/badge/License-MIT-green.svg

</div>
# GestureShape Matcher  
_A computer vision game controlled by hand gestures_  

## 🚀 Features
- **Pinch-to-drag** shapes using hand gestures
- Real-time **60+ FPS** hand tracking
- **4 Geometric shapes** (circle, square, rectangle, triangle)
- **Audio feedback** on successful matches

## 💻 Tech Stack
```mermaid
flowchart LR
    A[OpenCV] --> B(Image Processing)
    C[MediaPipe] --> D(Hand Tracking)
    E[PyGame] --> F(Game Interface)

Install requirements

pip install -r requirements.txt

Run the game
python game.py

📂 Files
text
assets/
   ├── shape_circle.png
   ├── shape_rectangle.png
   ├── shape_square.png
   ├── shape_triangle.png
   └── success.wav
game.py          # Main game logic
shape.py         # Shape generator
requirements.txt # Dependencies
