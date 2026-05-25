import cv2
import numpy as np
from datetime import timedelta

# Color mappings for classes (BGR format)
DEFAULT_COLORS = {
    "person": (255, 255, 0),  # Cyan
    "car": (75, 220, 75),     # Emerald Green
    "dog": (0, 165, 255)      # Orange
}

def draw_futuristic_box(img, x1, y1, x2, y2, color, label, score, line_thickness=2):
    """Draws a premium styled bounding box with corner brackets and a label tag."""
    h, w, _ = img.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    # Draw a thin bounding box line
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    
    # Draw thicker corner brackets for a futuristic/HUD look
    corner_len = min(20, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))
    # Top-Left corner
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, line_thickness, lineType=cv2.LINE_AA)
    # Top-Right corner
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, line_thickness, lineType=cv2.LINE_AA)
    # Bottom-Left corner
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, line_thickness, lineType=cv2.LINE_AA)
    # Bottom-Right corner
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, line_thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, line_thickness, lineType=cv2.LINE_AA)
    
    # Draw a tag background above or inside the box
    tag_text = f"{label} {int(score * 100)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    
    (tw, th), baseline = cv2.getTextSize(tag_text, font, font_scale, font_thickness)
    tag_y = y1 - 5 if y1 - 5 - th > 0 else y1 + th + 5
    tag_x = x1
    
    # Tag background
    cv2.rectangle(img, (tag_x, tag_y - th - 3), (tag_x + tw + 6, tag_y + baseline), color, -1)
    # Tag text (black text for yellow/orange backgrounds, white for blue/darker colors)
    text_color = (255, 255, 255) if color == (255, 0, 0) else (0, 0, 0)
    cv2.putText(img, tag_text, (tag_x + 3, tag_y - 2), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

def draw_hud(img, counts, elapsed_time, total_time, model_name, device):
    """Draws a premium top-bar HUD containing current counts, time, and system details."""
    h, w, _ = img.shape
    
    # HUD Bar dimensions
    hud_h = 55
    hud_bg = img[0:hud_h, 0:w].copy()
    
    # Apply semi-transparent black overlay
    overlay = np.zeros_like(hud_bg)
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 20, 30), -1)
    hud_bg = cv2.addWeighted(hud_bg, 0.4, overlay, 0.6, 0)
    
    # Write overlay back
    img[0:hud_h, 0:w] = hud_bg
    
    # Draw separation line
    cv2.line(img, (0, hud_h), (w, hud_h), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    
    # Text styles
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. Live Detections on the Left
    text_y = 33
    start_x = 20
    
    # People (Cyan)
    cv2.circle(img, (start_x, text_y - 6), 5, (255, 255, 0), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"People: {counts.get('person', 0)}", (start_x + 12, text_y), font, 0.5, (220, 240, 255), 1, lineType=cv2.LINE_AA)
    
    # Cars (Green)
    start_x += 130
    cv2.circle(img, (start_x, text_y - 6), 5, (75, 220, 75), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"Cars: {counts.get('car', 0)}", (start_x + 12, text_y), font, 0.5, (220, 240, 255), 1, lineType=cv2.LINE_AA)
    
    # Dogs (Orange)
    start_x += 110
    cv2.circle(img, (start_x, text_y - 6), 5, (0, 165, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"Dogs: {counts.get('dog', 0)}", (start_x + 12, text_y), font, 0.5, (220, 240, 255), 1, lineType=cv2.LINE_AA)
    
    # 2. Time & Progress in the Center/Right
    time_str = f"{str(timedelta(seconds=int(elapsed_time)))} / {str(timedelta(seconds=int(total_time)))}"
    (tw, _), _ = cv2.getTextSize(time_str, font, 0.5, 1)
    time_x = (w - tw) // 2
    cv2.putText(img, time_str, (time_x, text_y), font, 0.5, (200, 200, 200), 1, lineType=cv2.LINE_AA)
    
    # 3. Model & Device info on the Right
    info_str = f"Model: {model_name} | Device: {device.upper()}"
    (iw, _), _ = cv2.getTextSize(info_str, font, 0.4, 1)
    cv2.putText(img, info_str, (w - iw - 20, text_y), font, 0.45, (140, 150, 160), 1, lineType=cv2.LINE_AA)

def draw_overlays(frame, detections, counts, elapsed_time, total_time, model_name, device, class_colors=None):
    """Draws bounding boxes and HUD overlay on the given frame.
    
    Args:
        frame: The input frame (numpy array) to draw on.
        detections: List of detection dicts containing 'label', 'score', and 'box'.
        counts: Dictionary of counts for active classes.
        elapsed_time: Current time in seconds.
        total_time: Total video duration in seconds.
        model_name: Name of the active model.
        device: Active device string (e.g. 'cuda', 'cpu').
        class_colors: Optional dictionary mapping class names to BGR colors.
        
    Returns:
        The annotated frame (numpy array).
    """
    annotated_frame = frame.copy()
    colors = class_colors if class_colors is not None else DEFAULT_COLORS
    
    # Draw detections
    for det in detections:
        lbl = det["label"]
        color = colors.get(lbl, (128, 128, 128))  # Default to gray if class not mapped
        box = det["box"]
        draw_futuristic_box(
            annotated_frame,
            box[0],
            box[1],
            box[2],
            box[3],
            color,
            lbl.capitalize(),
            det["score"]
        )
        
    # Draw HUD
    draw_hud(annotated_frame, counts, elapsed_time, total_time, model_name, device)
    
    return annotated_frame
