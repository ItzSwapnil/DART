"""
DART UI Theme System - Centralized styling for the Tkinter interface.
Provides colors, typography, spacing, and custom style definitions.
"""
from tkinter import ttk

# =============================================================================
# COLOR PALETTE
# =============================================================================

class Colors:
    """Color definitions for DART UI."""

    # Primary colors
    PRIMARY = "#6366F1"       # Indigo - main accent
    PRIMARY_DARK = "#4F46E5"  # Darker indigo
    PRIMARY_LIGHT = "#818CF8" # Lighter indigo

    # Status colors
    SUCCESS = "#10B981"       # Emerald green
    SUCCESS_LIGHT = "#34D399"
    ERROR = "#EF4444"         # Red
    ERROR_LIGHT = "#F87171"
    WARNING = "#F59E0B"       # Amber
    WARNING_LIGHT = "#FBBF24"
    INFO = "#3B82F6"          # Blue
    INFO_LIGHT = "#60A5FA"

    # Neutral colors
    WHITE = "#FFFFFF"
    BLACK = "#000000"

    # Dark theme
    DARK_BG = "#1A1B26"           # Main background
    DARK_BG_SECONDARY = "#24283B" # Card/panel background
    DARK_BG_TERTIARY = "#2F3549"  # Hover/active states
    DARK_BORDER = "#414868"       # Borders
    DARK_TEXT = "#C0CAF5"         # Primary text
    DARK_TEXT_MUTED = "#565F89"   # Secondary text

    # Light theme
    LIGHT_BG = "#F8FAFC"
    LIGHT_BG_SECONDARY = "#FFFFFF"
    LIGHT_BG_TERTIARY = "#F1F5F9"
    LIGHT_BORDER = "#E2E8F0"
    LIGHT_TEXT = "#1E293B"
    LIGHT_TEXT_MUTED = "#64748B"

    # Trading colors
    PROFIT = "#10B981"    # Green for gains
    LOSS = "#EF4444"      # Red for losses
    NEUTRAL = "#6B7280"   # Gray for neutral


# =============================================================================
# TYPOGRAPHY
# =============================================================================

class Fonts:
    """Font definitions for DART UI."""

    # Font families
    FAMILY = "Segoe UI"
    FAMILY_MONO = "Consolas"

    # Font sizes
    SIZE_XS = 9
    SIZE_SM = 10
    SIZE_MD = 11
    SIZE_LG = 13
    SIZE_XL = 16
    SIZE_2XL = 20
    SIZE_3XL = 26

    # Common font tuples
    HEADING_LARGE = (FAMILY, SIZE_2XL, "bold")
    HEADING = (FAMILY, SIZE_XL, "bold")
    HEADING_SMALL = (FAMILY, SIZE_LG, "bold")
    BODY = (FAMILY, SIZE_MD)
    BODY_BOLD = (FAMILY, SIZE_MD, "bold")
    BODY_SMALL = (FAMILY, SIZE_SM)
    CAPTION = (FAMILY, SIZE_XS)
    MONO = (FAMILY_MONO, SIZE_MD)

    # Metric fonts
    BALANCE = (FAMILY, SIZE_2XL, "bold")
    METRIC_VALUE = (FAMILY, SIZE_LG, "bold")
    METRIC_LABEL = (FAMILY, SIZE_SM)


# =============================================================================
# SPACING
# =============================================================================

class Spacing:
    """Spacing constants for consistent layout."""

    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32

    # Padding tuples (horizontal, vertical)
    CARD = (LG, MD)
    SECTION = (XL, LG)
    BUTTON = (LG, SM)


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

def configure_styles(root, theme="dark"):
    """
    Configure ttk styles for the application.

    Args:
        root: Tkinter root window
        theme: "dark" or "light"
    """
    style = ttk.Style(root)

    # Determine colors based on theme
    if theme == "dark":
        bg_secondary = Colors.DARK_BG_SECONDARY
        text = Colors.DARK_TEXT
        text_muted = Colors.DARK_TEXT_MUTED
    else:
        bg_secondary = Colors.LIGHT_BG_SECONDARY
        text = Colors.LIGHT_TEXT
        text_muted = Colors.LIGHT_TEXT_MUTED

    # Configure Card style (for LabelFrames with card appearance)
    style.configure(
        "Card.TLabelframe",
        background=bg_secondary,
        borderwidth=1,
        relief="solid",
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=bg_secondary,
        foreground=text,
        font=Fonts.HEADING_SMALL,
    )

    # Primary action button
    style.configure(
        "Primary.TButton",
        font=Fonts.BODY_BOLD,
        padding=Spacing.BUTTON,
    )

    # Success button (green)
    style.configure(
        "Success.TButton",
        font=Fonts.BODY_BOLD,
        padding=Spacing.BUTTON,
    )

    # Danger button (red)
    style.configure(
        "Danger.TButton",
        font=Fonts.BODY_BOLD,
        padding=Spacing.BUTTON,
    )

    # Large metric label
    style.configure(
        "Metric.TLabel",
        font=Fonts.METRIC_VALUE,
        foreground=text,
    )

    # Metric label header
    style.configure(
        "MetricLabel.TLabel",
        font=Fonts.METRIC_LABEL,
        foreground=text_muted,
    )

    # Balance display (extra large)
    style.configure(
        "Balance.TLabel",
        font=Fonts.BALANCE,
        foreground=Colors.WHITE,
    )

    # Profit label
    style.configure(
        "Profit.TLabel",
        font=Fonts.METRIC_VALUE,
        foreground=Colors.PROFIT,
    )

    # Loss label
    style.configure(
        "Loss.TLabel",
        font=Fonts.METRIC_VALUE,
        foreground=Colors.LOSS,
    )

    # Status labels
    style.configure(
        "StatusSuccess.TLabel",
        font=Fonts.BODY_BOLD,
        foreground=Colors.SUCCESS,
    )

    style.configure(
        "StatusError.TLabel",
        font=Fonts.BODY_BOLD,
        foreground=Colors.ERROR,
    )

    style.configure(
        "StatusWarning.TLabel",
        font=Fonts.BODY_BOLD,
        foreground=Colors.WARNING,
    )

    style.configure(
        "StatusInfo.TLabel",
        font=Fonts.BODY_BOLD,
        foreground=Colors.INFO,
    )

    # Heading styles
    style.configure(
        "Heading.TLabel",
        font=Fonts.HEADING,
        foreground=text,
    )

    style.configure(
        "HeadingSmall.TLabel",
        font=Fonts.HEADING_SMALL,
        foreground=text,
    )

    # Muted/secondary text
    style.configure(
        "Muted.TLabel",
        font=Fonts.BODY_SMALL,
        foreground=text_muted,
    )

    return style


def get_status_color(status: str) -> str:
    """Get color based on status type."""
    status_lower = status.lower()

    if any(word in status_lower for word in ["success", "profit", "win", "connected", "trained", "ready"]):
        return Colors.SUCCESS
    elif any(word in status_lower for word in ["error", "fail", "loss", "disconnect"]):
        return Colors.ERROR
    elif any(word in status_lower for word in ["warning", "pause", "wait"]):
        return Colors.WARNING
    elif any(word in status_lower for word in ["trading", "running", "active"]):
        return Colors.INFO
    else:
        return Colors.DARK_TEXT_MUTED


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a number as currency."""
    if currency == "USD":
        prefix = "$"
    elif currency == "EUR":
        prefix = "€"
    elif currency == "GBP":
        prefix = "£"
    else:
        prefix = f"{currency} "

    if value >= 0:
        return f"{prefix}{value:,.2f}"
    else:
        return f"-{prefix}{abs(value):,.2f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format a number as percentage."""
    return f"{value * 100:.{decimal_places}f}%"
