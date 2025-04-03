import asyncio
import threading
import tkinter as tk
from tkinter import ttk
import sv_ttk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf

from api.deriv_client import DerivClient
from ui.chart_styles import get_dark_style
from utils.timeframe import get_granularity_mapping
from config.settings import DERIV_APP_ID, DEFAULT_CANDLE_COUNT

class DerivApp:
    """Main application class for the Deriv Markets Viewer."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Deriv Markets Viewer")
        self.client = DerivClient(app_id=DERIV_APP_ID)
        
        # Apply the Sun Valley theme
        sv_ttk.set_theme("dark")
        
        # Setup UI components
        self._setup_ui()
        
        # Start the asyncio event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.start_async_loop, daemon=True).start()
        
        # Schedule the population of markets
        asyncio.run_coroutine_threadsafe(self.populate_markets(), self.loop)
    
    def _setup_ui(self):
        """Set up the user interface components."""
        # Frame for controls
        self.control_frame = ttk.Frame(self.root, padding=(10, 5))
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Market selection
        self.market_label = ttk.Label(self.control_frame, text="Select a Market:")
        self.market_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.market_var = tk.StringVar()
        self.market_combobox = ttk.Combobox(self.control_frame, textvariable=self.market_var, state='readonly')
        self.market_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.market_combobox.bind('<<ComboboxSelected>>', self.on_selection_change)
        
        # Timeframe selection
        self.timeframe_label = ttk.Label(self.control_frame, text="Select Timeframe:")
        self.timeframe_label.grid(row=0, column=2, padx=5, pady=5)
        
        self.timeframe_var = tk.StringVar()
        self.timeframe_combobox = ttk.Combobox(self.control_frame, textvariable=self.timeframe_var, state='readonly')
        self.timeframe_combobox['values'] = list(get_granularity_mapping().keys())
        self.timeframe_combobox.grid(row=0, column=3, padx=5, pady=5)
        self.timeframe_combobox.bind('<<ComboboxSelected>>', self.on_selection_change)
        self.timeframe_combobox.current(0)  # Default to DEFAULT_TIMEFRAME
        
        # Frame for chart display
        self.chart_frame = ttk.Frame(self.root, padding=(10, 5))
        self.chart_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Dictionary to store market symbols
        self.symbols_dict = {}
    
    def start_async_loop(self):
        """Start the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    async def populate_markets(self):
        """Fetch and display active markets."""
        self.symbols_dict = await self.client.get_active_symbols()
        
        # Update the Combobox in the main thread
        self.root.after(0, self.update_market_combobox)
    
    def update_market_combobox(self):
        """Update the Combobox with market data."""
        market_names = list(self.symbols_dict.keys())
        self.market_combobox['values'] = market_names
        if market_names:
            self.market_combobox.current(0)  # Set default selection
    
    def on_selection_change(self, event):
        """Handle selection changes for market and timeframe."""
        market = self.market_var.get()
        timeframe = self.timeframe_var.get()
        if market and timeframe:
            symbol = self.symbols_dict[market]
            granularity = get_granularity_mapping()[timeframe]
            # Schedule the update_chart coroutine
            asyncio.run_coroutine_threadsafe(self.update_chart(symbol, granularity), self.loop)
    
    async def update_chart(self, symbol, granularity):
        """Fetch data and update the candlestick chart."""
        candles = await self.client.get_candles(symbol, granularity, count=DEFAULT_CANDLE_COUNT)
        # Update the chart in the main thread
        self.root.after(0, self.plot_candlestick_chart, candles)
    
    def plot_candlestick_chart(self, candles):
        """Plot candlestick chart with the given candle data."""
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
        
        if not candles:
            return
        
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['epoch'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                  inplace=True)
        
        fig, ax = mpf.plot(
            df,
            type='candle',
            style=get_dark_style(),
            returnfig=True
        )
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
