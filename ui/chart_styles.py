import mplfinance as mpf

def get_dark_style():
    """Returns a dark-themed style for mplfinance charts."""
    return mpf.make_mpf_style(
        base_mpl_style='dark_background',
        marketcolors=mpf.make_marketcolors(
            up='green',
            down='red',
            edge={'up': 'green', 'down': 'red'},
            wick={'up': 'green', 'down': 'red'},
            volume={'up': 'green', 'down': 'red'},
            ohlc={'up': 'green', 'down': 'red'}
        ),
        mavcolors=['#1f77b4', '#ff7f0e', '#2ca02c'],
        facecolor='#121212',
        gridcolor='#2A2A2A',
        gridstyle='--'
    )
