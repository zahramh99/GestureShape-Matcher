# GestureShape-Matcher
 A computer vision game that matches shapes using hand gestures.

A computer vision game that lets you match shapes using hand gestures. Pinch to grab and drag shapes to their matching targets!

## Features

- ✋ **Hand gesture control** using MediaPipe
- 🎯 **4 geometric shapes** to match (circle, square, rectangle, triangle)
- 🔊 **Audio feedback** on successful matches
- 🖥️ Real-time **OpenCV** visualization
- 🎮 Simple **PyGame** integration

## Requirements

- Python 3.8+
- Webcam

# Project Structure
gesture-shape-matcher/
├── game.py               # Main game logic
├── shape.py              # Shape generator utility
├── requirements.txt      # Dependencies
├── README.md             # This file
├── LICENSE               # MIT License
└── assets/
    ├── shape_circle.png
    ├── shape_square.png
    ├── shape_rectangle.png
    ├── shape_triangle.png
    └── success.wav
