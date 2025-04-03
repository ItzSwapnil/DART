"""
DART - Deep Adaptive Reinforcement Trader
Main entry point for the application.
"""
import tkinter as tk
from ui.app import DerivApp

def main():
    """Initialize and run the application."""
    root = tk.Tk()
    app = DerivApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
