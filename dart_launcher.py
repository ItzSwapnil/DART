"""
DART Launcher - Choose Your Interface
Provides options to launch either the modern web dashboard or classic desktop interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import threading
import webbrowser
import time

class DARTLauncher:
    """Launcher for DART with multiple interface options."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DART Launcher - Choose Your Experience")
        self.root.geometry("800x600")
        self.root.configure(bg='#0f0f23')
        
        # Center the window
        self.center_window()
        
        # Setup UI
        self.setup_ui()
        
        # Store processes
        self.streamlit_process = None
        
    def center_window(self):
        """Center the launcher window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_ui(self):
        """Setup the launcher UI."""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#0f0f23')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🎯 DART - Deep Adaptive Reinforcement Trader",
            font=('Arial', 24, 'bold'),
            fg='#3b82f6',
            bg='#0f0f23'
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = tk.Label(
            main_frame,
            text="Professional AI Trading Platform",
            font=('Arial', 14),
            fg='#94a3b8',
            bg='#0f0f23'
        )
        subtitle_label.pack(pady=(0, 40))
        
        # Interface selection frame
        selection_frame = tk.Frame(main_frame, bg='#0f0f23')
        selection_frame.pack(fill=tk.BOTH, expand=True)
        
        # Modern Web Dashboard Option
        self.create_option_card(
            selection_frame,
            "🚀 Modern Web Dashboard",
            "Professional web-based interface with real-time charts, AI insights, and modern UI",
            [
                "• Real-time interactive charts with Plotly",
                "• Modern responsive design",
                "• AI market sentiment analysis",
                "• Advanced performance analytics",
                "• Professional trading interface",
                "• Mobile-friendly responsive layout"
            ],
            self.launch_web_dashboard,
            row=0, column=0
        )
        
        # Classic Desktop Interface Option
        self.create_option_card(
            selection_frame,
            "🖥️ Classic Desktop Interface",
            "Traditional desktop application with comprehensive trading tools",
            [
                "• Native desktop performance",
                "• Complete trading functionality",
                "• Real-time market data",
                "• AI model training & deployment",
                "• Automated trading system",
                "• Familiar desktop experience"
            ],
            self.launch_desktop_app,
            row=0, column=1
        )
        
        # Configure grid weights
        selection_frame.columnconfigure(0, weight=1)
        selection_frame.columnconfigure(1, weight=1)
        selection_frame.rowconfigure(0, weight=1)
        
        # Footer
        footer_frame = tk.Frame(main_frame, bg='#0f0f23')
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(40, 0))
        
        footer_label = tk.Label(
            footer_frame,
            text="⚡ Both interfaces connect to the same AI trading engine\n🔒 Your settings and models are shared between interfaces",
            font=('Arial', 10),
            fg='#6b7280',
            bg='#0f0f23',
            justify=tk.CENTER
        )
        footer_label.pack()\n    \n    def create_option_card(self, parent, title, description, features, command, row, column):\n        \"\"\"Create an option card for interface selection.\"\"\"\n        # Card frame\n        card_frame = tk.Frame(\n            parent,\n            bg='#1e1e3f',\n            relief=tk.RAISED,\n            borderwidth=2\n        )\n        card_frame.grid(row=row, column=column, padx=20, pady=20, sticky='nsew')\n        \n        # Card content\n        content_frame = tk.Frame(card_frame, bg='#1e1e3f')\n        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)\n        \n        # Title\n        title_label = tk.Label(\n            content_frame,\n            text=title,\n            font=('Arial', 18, 'bold'),\n            fg='#ffffff',\n            bg='#1e1e3f'\n        )\n        title_label.pack(anchor='w', pady=(0, 10))\n        \n        # Description\n        desc_label = tk.Label(\n            content_frame,\n            text=description,\n            font=('Arial', 11),\n            fg='#94a3b8',\n            bg='#1e1e3f',\n            wraplength=300,\n            justify=tk.LEFT\n        )\n        desc_label.pack(anchor='w', pady=(0, 15))\n        \n        # Features\n        features_label = tk.Label(\n            content_frame,\n            text='\\n'.join(features),\n            font=('Arial', 10),\n            fg='#10b981',\n            bg='#1e1e3f',\n            justify=tk.LEFT\n        )\n        features_label.pack(anchor='w', pady=(0, 20))\n        \n        # Launch button\n        launch_button = tk.Button(\n            content_frame,\n            text=\"Launch Interface\",\n            font=('Arial', 12, 'bold'),\n            fg='#ffffff',\n            bg='#3b82f6',\n            activebackground='#2563eb',\n            activeforeground='#ffffff',\n            cursor='hand2',\n            command=command,\n            padx=20,\n            pady=10\n        )\n        launch_button.pack(side=tk.BOTTOM, fill=tk.X)\n        \n        # Hover effects\n        def on_enter(e):\n            card_frame.configure(bg='#2a2a5a')\n            content_frame.configure(bg='#2a2a5a')\n            title_label.configure(bg='#2a2a5a')\n            desc_label.configure(bg='#2a2a5a')\n            features_label.configure(bg='#2a2a5a')\n        \n        def on_leave(e):\n            card_frame.configure(bg='#1e1e3f')\n            content_frame.configure(bg='#1e1e3f')\n            title_label.configure(bg='#1e1e3f')\n            desc_label.configure(bg='#1e1e3f')\n            features_label.configure(bg='#1e1e3f')\n        \n        card_frame.bind('<Enter>', on_enter)\n        card_frame.bind('<Leave>', on_leave)\n        content_frame.bind('<Enter>', on_enter)\n        content_frame.bind('<Leave>', on_leave)\n    \n    def launch_web_dashboard(self):\n        \"\"\"Launch the modern web dashboard.\"\"\"\n        try:\n            # Show loading message\n            loading_window = self.show_loading_window(\"Starting Modern Web Dashboard...\")\n            \n            def start_streamlit():\n                try:\n                    # Start Streamlit in a separate process\n                    dashboard_path = os.path.join(os.path.dirname(__file__), 'modern_dashboard.py')\n                    \n                    self.streamlit_process = subprocess.Popen([\n                        sys.executable, '-m', 'streamlit', 'run', dashboard_path,\n                        '--server.port=8501',\n                        '--server.headless=true',\n                        '--browser.gatherUsageStats=false'\n                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n                    \n                    # Wait a bit for Streamlit to start\n                    time.sleep(3)\n                    \n                    # Open browser\n                    webbrowser.open('http://localhost:8501')\n                    \n                    # Close loading window\n                    self.root.after(0, loading_window.destroy)\n                    \n                    # Show success message\n                    self.root.after(0, lambda: messagebox.showinfo(\n                        \"Dashboard Launched\",\n                        \"Modern Web Dashboard is now running!\\n\\nURL: http://localhost:8501\\n\\nThe dashboard will open in your default browser.\"\n                    ))\n                    \n                except Exception as e:\n                    self.root.after(0, loading_window.destroy)\n                    self.root.after(0, lambda: messagebox.showerror(\n                        \"Launch Error\",\n                        f\"Failed to launch web dashboard:\\n{str(e)}\"\n                    ))\n            \n            # Start in separate thread\n            threading.Thread(target=start_streamlit, daemon=True).start()\n            \n        except Exception as e:\n            messagebox.showerror(\"Launch Error\", f\"Failed to launch web dashboard: {str(e)}\")\n    \n    def launch_desktop_app(self):\n        \"\"\"Launch the classic desktop interface.\"\"\"\n        try:\n            # Show loading message\n            loading_window = self.show_loading_window(\"Starting Classic Desktop Interface...\")\n            \n            def start_desktop():\n                try:\n                    # Import and start the desktop app\n                    from ui.app import DerivApp\n                    \n                    # Close loading window\n                    self.root.after(0, loading_window.destroy)\n                    \n                    # Close launcher\n                    self.root.after(0, self.root.destroy)\n                    \n                    # Start desktop app\n                    desktop_root = tk.Tk()\n                    app = DerivApp(desktop_root)\n                    desktop_root.mainloop()\n                    \n                except Exception as e:\n                    self.root.after(0, loading_window.destroy)\n                    self.root.after(0, lambda: messagebox.showerror(\n                        \"Launch Error\",\n                        f\"Failed to launch desktop interface:\\n{str(e)}\"\n                    ))\n            \n            # Start in separate thread\n            threading.Thread(target=start_desktop, daemon=True).start()\n            \n        except Exception as e:\n            messagebox.showerror(\"Launch Error\", f\"Failed to launch desktop interface: {str(e)}\")\n    \n    def show_loading_window(self, message):\n        \"\"\"Show a loading window.\"\"\"\n        loading = tk.Toplevel(self.root)\n        loading.title(\"Loading...\")\n        loading.geometry(\"400x150\")\n        loading.configure(bg='#0f0f23')\n        loading.resizable(False, False)\n        \n        # Center the loading window\n        loading.transient(self.root)\n        loading.grab_set()\n        \n        # Loading content\n        loading_frame = tk.Frame(loading, bg='#0f0f23')\n        loading_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)\n        \n        loading_label = tk.Label(\n            loading_frame,\n            text=message,\n            font=('Arial', 12),\n            fg='#ffffff',\n            bg='#0f0f23'\n        )\n        loading_label.pack(pady=(20, 10))\n        \n        # Progress bar\n        progress = ttk.Progressbar(\n            loading_frame,\n            mode='indeterminate'\n        )\n        progress.pack(fill=tk.X, pady=10)\n        progress.start()\n        \n        loading.update()\n        return loading\n    \n    def on_closing(self):\n        \"\"\"Handle application closing.\"\"\"\n        if self.streamlit_process:\n            try:\n                self.streamlit_process.terminate()\n            except:\n                pass\n        \n        self.root.destroy()\n    \n    def run(self):\n        \"\"\"Run the launcher.\"\"\"\n        self.root.protocol(\"WM_DELETE_WINDOW\", self.on_closing)\n        self.root.mainloop()\n\nif __name__ == \"__main__\":\n    launcher = DARTLauncher()\n    launcher.run()"
