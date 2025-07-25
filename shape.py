from PIL import Image, ImageDraw
from typing import Callable

class ShapeGenerator:
    """Generates shape images for the matching game."""
    
    SHAPE_COLORS = {
        "circle": "red",
        "square": "blue",
        "rectangle": "green",
        "triangle": "orange"
    }
    
    def __init__(self):
        """Initialize the shape generator."""
        self.size = (100, 100)  # All shapes will be 100x100 pixels
        
    def generate_all_shapes(self):
        """Generate all default shapes for the game."""
        shapes = [
            ("circle", self._draw_circle),
            ("square", self._draw_square),
            ("rectangle", self._draw_rectangle),
            ("triangle", self._draw_triangle)
        ]
        
        for name, draw_func in shapes:
            self.generate_shape(name, draw_func)
            
    def generate_shape(self, name: str, draw_function: Callable):
        """
        Generate a single shape image.
        
        Args:
            name: Name of the shape (used for filename)
            draw_function: Function that implements the drawing logic
        """
        try:
            # Create transparent image
            img = Image.new("RGBA", self.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw the shape
            draw_function(draw)
            
            # Save the image
            filename = f"shape_{name}.png"
            img.save(filename)
            print(f"✅ Saved: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to generate shape {name}: {e}")
            
    def _draw_circle(self, draw: ImageDraw):
        """Draw a circle shape."""
        draw.ellipse(
            (10, 10, 90, 90),
            fill=self.SHAPE_COLORS["circle"],
            outline="black",
            width=3
        )
        
    def _draw_square(self, draw: ImageDraw):
        """Draw a square shape."""
        draw.rectangle(
            (10, 10, 90, 90),
            fill=self.SHAPE_COLORS["square"],
            outline="black",
            width=3
        )
        
    def _draw_rectangle(self, draw: ImageDraw):
        """Draw a rectangle shape."""
        draw.rectangle(
            (10, 25, 90, 75),
            fill=self.SHAPE_COLORS["rectangle"],
            outline="black",
            width=3
        )
        
    def _draw_triangle(self, draw: ImageDraw):
        """Draw a triangle shape."""
        draw.polygon(
            [(50, 10), (90, 90), (10, 90)],
            fill=self.SHAPE_COLORS["triangle"],
            outline="black"
        )

if __name__ == "__main__":
    generator = ShapeGenerator()
    generator.generate_all_shapes()