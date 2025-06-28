import mplfinance as mpf

from config.settings import CHART_STYLES


def get_dark_style():
    """Returns a dark-themed style for mplfinance charts."""
    dark_style = CHART_STYLES.get("dark", {})
    return mpf.make_mpf_style(
        base_mpl_style=dark_style.get("base_mpl_style", "dark_background"),
        marketcolors=mpf.make_marketcolors(
            up=dark_style.get("up_color", "green"),
            down=dark_style.get("down_color", "red"),
            edge={
                "up": dark_style.get("up_color", "green"),
                "down": dark_style.get("down_color", "red"),
            },
            wick={
                "up": dark_style.get("up_color", "green"),
                "down": dark_style.get("down_color", "red"),
            },
            volume={
                "up": dark_style.get("up_color", "green"),
                "down": dark_style.get("down_color", "red"),
            },
            ohlc={
                "up": dark_style.get("up_color", "green"),
                "down": dark_style.get("down_color", "red"),
            },
        ),
        mavcolors=dark_style.get("mavcolors", ["#1f77b4", "#ff7f0e", "#2ca02c"]),
        facecolor=dark_style.get("facecolor", "#121212"),
        gridcolor=dark_style.get("gridcolor", "#2A2A2A"),
        gridstyle=dark_style.get("gridstyle", "--"),
    )


def get_light_style():
    """Returns a light-themed style for mplfinance charts."""
    light_style = CHART_STYLES.get("light", {})
    return mpf.make_mpf_style(
        base_mpl_style=light_style.get("base_mpl_style", "default"),
        marketcolors=mpf.make_marketcolors(
            up=light_style.get("up_color", "green"),
            down=light_style.get("down_color", "red"),
            edge={
                "up": light_style.get("up_color", "green"),
                "down": light_style.get("down_color", "red"),
            },
            wick={
                "up": light_style.get("up_color", "green"),
                "down": light_style.get("down_color", "red"),
            },
            volume={
                "up": light_style.get("up_color", "green"),
                "down": light_style.get("down_color", "red"),
            },
            ohlc={
                "up": light_style.get("up_color", "green"),
                "down": light_style.get("down_color", "red"),
            },
        ),
        mavcolors=light_style.get("mavcolors", ["#1f77b4", "#ff7f0e", "#2ca02c"]),
        facecolor=light_style.get("facecolor", "white"),
        gridcolor=light_style.get("gridcolor", "#E6E6E6"),
        gridstyle=light_style.get("gridstyle", "-"),
    )


def get_chart_style(theme="dark"):
    """Returns the appropriate chart style based on the theme."""
    if theme.lower() == "light":
        return get_light_style()
    return get_dark_style()
