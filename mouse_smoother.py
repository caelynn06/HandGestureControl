# mouse_smoother.py
"""
Mouse smoothing and movement control
This file contains classes that make mouse movement smooth and natural.
Without smoothing, the cursor would be jittery and hard to control.
"""

# Import required libraries
from collections import deque  # Double-ended queue for efficient buffer operations
import pyautogui  # For controlling the mouse cursor
import numpy as np  # For mathematical operations (not used here but common in similar code)


class MouseSmoothing:
    """
    Smooth mouse movement using moving average and exponential smoothing.
    This class stores recent mouse positions and calculates a smooth average.
    
    Two smoothing techniques are combined:
    1. Moving Average - average of last N positions
    2. Exponential Smoothing - weighted average favoring recent positions
    """
    
    def __init__(self, buffer_size=5, exponential_weight=0.3):
        """
        Initialize the smoothing system.
        
        Args:
            buffer_size: How many recent positions to store (default: 5)
                        Larger = smoother but more lag
                        Smaller = more responsive but jittery
            exponential_weight: Weight for new positions (0.0 to 1.0, default: 0.3)
                               Higher = more responsive
                               Lower = smoother
        """
        # Store the buffer size
        self.buffer_size = buffer_size
        
        # Store the exponential smoothing weight
        self.exp_weight = exponential_weight
        
        # Create a deque (double-ended queue) to store positions
        # maxlen=buffer_size means old positions automatically drop off when full
        # This is more efficient than a regular list for this purpose
        self.position_buffer = deque(maxlen=buffer_size)
        
        # Store the current smoothed position
        # None at start because we haven't received any positions yet
        self.smooth_x = None
        self.smooth_y = None
        
    def add_position(self, x, y):
        """
        Add a new mouse position to the smoothing buffer.
        This should be called every frame with the raw cursor position.
        
        Args:
            x: Target x coordinate (in screen pixels)
            y: Target y coordinate (in screen pixels)
        """
        # Append the new position to our buffer as a tuple (x, y)
        # If buffer is full, the oldest position is automatically removed
        self.position_buffer.append((x, y))
        
    def get_smoothed_position(self):
        """
        Calculate and return the smoothed mouse position.
        This uses both moving average and exponential smoothing.
        
        Returns:
            Tuple (smooth_x, smooth_y) - the smoothed coordinates
            or (None, None) if buffer is empty
        """
        # Check if buffer has any positions
        if not self.position_buffer:
            # No positions stored yet, return None
            return None, None
        
        # STEP 1: Calculate Moving Average
        # Sum all x-coordinates in buffer
        # pos[0] gets the x-coordinate from each position tuple
        sum_x = sum(pos[0] for pos in self.position_buffer)
        
        # Calculate average x by dividing sum by number of positions
        avg_x = sum_x / len(self.position_buffer)
        
        # Do the same for y-coordinates
        # pos[1] gets the y-coordinate from each position tuple
        sum_y = sum(pos[1] for pos in self.position_buffer)
        avg_y = sum_y / len(self.position_buffer)
        
        # STEP 2: Apply Exponential Smoothing
        # This gives extra weight to the previous smoothed position
        
        # Check if this is the first position we're calculating
        if self.smooth_x is None:
            # No previous smooth position, so just use the average
            self.smooth_x = avg_x
            self.smooth_y = avg_y
        else:
            # We have a previous smooth position, so blend it with new average
            # Formula: new_smooth = (weight × new_average) + ((1-weight) × old_smooth)
            # This creates momentum - the cursor "remembers" where it was going
            self.smooth_x = self.exp_weight * avg_x + (1 - self.exp_weight) * self.smooth_x
            self.smooth_y = self.exp_weight * avg_y + (1 - self.exp_weight) * self.smooth_y
        
        # Return the final smoothed coordinates
        return self.smooth_x, self.smooth_y


class MouseController:
    """
    Handle mouse movement and clicking with advanced smoothing.
    This class manages the entire pipeline from hand position to cursor position.
    """
    
    def __init__(self, screen_width, screen_height, edge_margin=0.15):
        """
        Initialize the mouse controller.
        
        Args:
            screen_width: Width of screen in pixels
            screen_height: Height of screen in pixels
            edge_margin: Fraction of screen to apply edge dampening (default: 0.15)
                        0.15 means outer 15% on each side has reduced sensitivity
        """
        # Store screen dimensions
        self.screen_w = screen_width
        self.screen_h = screen_height
        
        # Store edge margin for dampening
        self.edge_margin = edge_margin
        
        # Create a MouseSmoothing object with specific settings
        # buffer_size=7 stores last 7 positions
        # exponential_weight=0.25 gives 25% weight to new positions
        self.smoother = MouseSmoothing(buffer_size=7, exponential_weight=0.25)
        
    def process_movement(self, hand_x, hand_y, frame_width, frame_height):
        """
        Process hand position and move mouse cursor smoothly.
        This is the main function that gets called every frame.
        
        Args:
            hand_x: X position of hand landmark in camera frame
            hand_y: Y position of hand landmark in camera frame
            frame_width: Width of camera frame in pixels
            frame_height: Height of camera frame in pixels
        """
        # STEP 1: Normalize coordinates to 0-1 range
        # This makes the values independent of camera resolution
        # Divide hand position by frame size to get fraction (0.0 to 1.0)
        norm_x = hand_x / frame_width   # 0.0 = left edge, 1.0 = right edge
        norm_y = hand_y / frame_height  # 0.0 = top edge, 1.0 = bottom edge
        
        # STEP 2: Apply edge dampening
        # Reduce sensitivity near screen edges to prevent overshoot
        # This makes it easier to reach screen corners and edges
        norm_x = self._apply_edge_dampening(norm_x)
        norm_y = self._apply_edge_dampening(norm_y)
        
        # STEP 3: Map to screen coordinates
        # Multiply normalized values (0-1) by screen dimensions
        # This converts to actual pixel positions on screen
        target_x = norm_x * self.screen_w  # Calculate target x position
        target_y = norm_y * self.screen_h  # Calculate target y position
        
        # STEP 4: Add to smoothing buffer
        # Send the target position to our smoother
        self.smoother.add_position(target_x, target_y)
        
        # STEP 5: Get smoothed position
        # Retrieve the smoothed coordinates from the buffer
        smooth_x, smooth_y = self.smoother.get_smoothed_position()
        
        # STEP 6: Move the actual mouse cursor
        # Only move if we have valid smoothed coordinates
        if smooth_x is not None and smooth_y is not None:
            # Try to move the mouse
            try:
                # Move cursor to smoothed position
                # duration=0 means move instantly (no pyautogui animation)
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            except:
                # If move fails (invalid coordinates, etc.), just skip
                # Using bare except is normally bad practice, but okay here
                # since we just want to ignore any movement errors
                pass
                
    def _apply_edge_dampening(self, value):
        """
        Reduce sensitivity near edges (0.0 or 1.0).
        This is a private helper method (indicated by _ prefix).
        
        Edge dampening prevents the cursor from overshooting at screen boundaries.
        Without this, it's very hard to reach screen corners precisely.
        
        Args:
            value: Normalized coordinate (0.0 to 1.0)
        
        Returns:
            Dampened coordinate value (still 0.0 to 1.0 range)
        """
        # Check if we're in the LEFT edge zone (less than edge_margin)
        # For example, if edge_margin=0.15, this catches values < 0.15
        if value < self.edge_margin:
            # Apply dampening formula for left edge
            # This compresses the left 15% of movement into a smaller range
            # Result: slower movement near left edge
            return value * (1 - self.edge_margin) / self.edge_margin
            
        # Check if we're in the RIGHT edge zone (greater than 1-edge_margin)
        # For example, if edge_margin=0.15, this catches values > 0.85
        elif value > (1 - self.edge_margin):
            # Apply dampening formula for right edge
            # This compresses the right 15% of movement into a smaller range
            # Result: slower movement near right edge
            return self.edge_margin + (value - (1 - self.edge_margin)) * (1 - self.edge_margin) / self.edge_margin
            
        # If we're not in an edge zone, return value unchanged
        # This is the center 70% of the screen (if edge_margin=0.15)
        return value