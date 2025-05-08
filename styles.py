# Color scheme
COLORS = {
    "primary": "#1f538d",
    "secondary": "#2d4059",
    "accent": "#ea5455",
    "background": "#212b38",
    "surface": "#2c3e50",
    "text": "#ffffff",
    "success": "#2ecc71",
    "error": "#e74c3c"
}

# Font configurations
FONTS = {
    "title": ("Helvetica", 24, "bold"),
    "subtitle": ("Helvetica", 16, "bold"),
    "body": ("Helvetica", 12),
    "button": ("Helvetica", 13, "bold")
}

# Widget styling
BUTTON_STYLE = {
    "corner_radius": 8,
    "border_width": 0,
    "text_color": "white",
    "hover": True,
    "fg_color": COLORS["primary"]
}

FRAME_STYLE = {
    "corner_radius": 10,
    "border_width": 2,
    "fg_color": COLORS["surface"],
    "border_color": COLORS["primary"]
}
