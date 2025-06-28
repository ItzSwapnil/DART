"""
DART Launcher - Choose Your Interface
Provides options to launch either the modern web dashboard or classic desktop interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import sys
import os
import threading
import webbrowser
import time
import datetime

class DARTLauncher:
    """Launcher for DART with multiple interface options."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DART Launcher - Choose Your Experience")
        self.root.geometry("1000x700")
        self.root.configure(bg='#0f0f23')
        
        # Center the window
        self.center_window()
        
        # Setup UI with tabs
        self.setup_ui()
        
        # Store processes and logs
        self.streamlit_process = None
        self.desktop_process = None
        self.log_buffer = []
        
    def center_window(self):
        """Center the launcher window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_ui(self):
        """Setup the launcher UI with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main launcher tab
        self.launcher_tab = tk.Frame(self.notebook, bg='#0f0f23')
        self.notebook.add(self.launcher_tab, text="🚀 Launcher")
        
        # Output logs tab
        self.logs_tab = tk.Frame(self.notebook, bg='#1a1a3e')
        self.notebook.add(self.logs_tab, text="📊 Output Logs")
        
        # Setup launcher content
        self.setup_launcher_tab()
        
        # Setup logs content
        self.setup_logs_tab()
    
    def setup_launcher_tab(self):
        """Setup the main launcher tab content."""
        # Main frame
        main_frame = tk.Frame(self.launcher_tab, bg='#0f0f23')
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
        footer_label.pack()
    
    def create_option_card(self, parent, title, description, features, command, row, column):
        """Create an option card for interface selection."""
        # Card frame
        card_frame = tk.Frame(
            parent,
            bg='#1e1e3f',
            relief=tk.RAISED,
            borderwidth=2
        )
        card_frame.grid(row=row, column=column, padx=20, pady=20, sticky='nsew')
        
        # Card content
        content_frame = tk.Frame(card_frame, bg='#1e1e3f')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            content_frame,
            text=title,
            font=('Arial', 18, 'bold'),
            fg='#ffffff',
            bg='#1e1e3f'
        )
        title_label.pack(anchor='w', pady=(0, 10))
        
        # Description
        desc_label = tk.Label(
            content_frame,
            text=description,
            font=('Arial', 11),
            fg='#94a3b8',
            bg='#1e1e3f',
            wraplength=300,
            justify=tk.LEFT
        )
        desc_label.pack(anchor='w', pady=(0, 15))
        
        # Features
        features_label = tk.Label(
            content_frame,
            text='\n'.join(features),
            font=('Arial', 10),
            fg='#10b981',
            bg='#1e1e3f',
            justify=tk.LEFT
        )
        features_label.pack(anchor='w', pady=(0, 20))
        
        # Launch button
        launch_button = tk.Button(
            content_frame,
            text="Launch Interface",
            font=('Arial', 12, 'bold'),
            fg='#ffffff',
            bg='#3b82f6',
            activebackground='#2563eb',
            activeforeground='#ffffff',
            cursor='hand2',
            command=command,
            padx=20,
            pady=10
        )
        launch_button.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Hover effects
        def on_enter(e):
            card_frame.configure(bg='#2a2a5a')
            content_frame.configure(bg='#2a2a5a')
            title_label.configure(bg='#2a2a5a')
            desc_label.configure(bg='#2a2a5a')
            features_label.configure(bg='#2a2a5a')
        
        def on_leave(e):
            card_frame.configure(bg='#1e1e3f')
            content_frame.configure(bg='#1e1e3f')
            title_label.configure(bg='#1e1e3f')
            desc_label.configure(bg='#1e1e3f')
            features_label.configure(bg='#1e1e3f')
        
        card_frame.bind('<Enter>', on_enter)
        card_frame.bind('<Leave>', on_leave)
        content_frame.bind('<Enter>', on_enter)
        content_frame.bind('<Leave>', on_leave)
    
    def launch_web_dashboard(self):
        """Launch the modern web dashboard."""
        try:
            self.log_message("🚀 Launching Modern Web Dashboard...", "web")
            
            # Show loading message
            loading_window = self.show_loading_window("Starting Modern Web Dashboard...")
            
            def start_streamlit():
                try:
                    self.log_message("📡 Starting Streamlit server...", "web")
                    # Start Streamlit in a separate process
                    dashboard_path = os.path.join(os.path.dirname(__file__), 'ui', 'modern_dashboard.py')
                    
                    self.streamlit_process = subprocess.Popen([
                        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
                        '--server.port=8501',
                        '--server.headless=true',
                        '--browser.gatherUsageStats=false'
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    self.log_message("⏳ Waiting for Streamlit to initialize...", "web")
                    
                    # Monitor Streamlit output
                    def monitor_streamlit():
                        while self.streamlit_process and self.streamlit_process.poll() is None:
                            try:
                                output = self.streamlit_process.stdout.readline()
                                if output:
                                    self.log_message(f"📡 {output.strip()}", "web")
                            except:
                                break
                    
                    # Start monitoring in background
                    threading.Thread(target=monitor_streamlit, daemon=True).start()
                    
                    # Wait a bit for Streamlit to start
                    time.sleep(4)
                    
                    self.log_message("🌐 Opening browser to http://localhost:8501", "web")
                    # Open browser
                    webbrowser.open('http://localhost:8501')
                    
                    # Close loading window
                    self.root.after(0, loading_window.destroy)
                    
                    self.log_message("✅ Web Dashboard launched successfully!", "web")
                    
                    # Show success message
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Dashboard Launched",
                        "Modern Web Dashboard is now running!\n\nURL: http://localhost:8501\n\nThe dashboard will open in your default browser."
                    ))
                    
                except Exception as e:
                    self.root.after(0, loading_window.destroy)
                    self.root.after(0, lambda: messagebox.showerror(
                        "Launch Error",
                        f"Failed to launch web dashboard:\n{str(e)}"
                    ))
            
            # Start in separate thread
            threading.Thread(target=start_streamlit, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch web dashboard: {str(e)}")
    
    def launch_desktop_app(self):
        """Launch the classic desktop interface."""
        try:
            self.log_message("🚀 Launching Classic Desktop Interface...", "desktop")
            
            # Import the desktop app first to check for issues
            from ui.app import DerivApp
            
            self.log_message("✅ Desktop module imported successfully", "desktop")
            
            # Show loading message briefly
            loading_window = self.show_loading_window("Starting Classic Desktop Interface...")
            
            def start_desktop():
                try:
                    self.log_message("🖥️ Initializing desktop interface...", "desktop")
                    
                    # Close loading window first
                    loading_window.destroy()
                    
                    self.log_message("🎯 Starting DART desktop application...", "desktop")
                    
                    # Close launcher (this will destroy log widgets)
                    self.root.destroy()
                    
                    # Start desktop app in main thread
                    desktop_root = tk.Tk()
                    app = DerivApp(desktop_root)
                    
                    # Don't try to log after launcher is destroyed
                    print("✅ Desktop interface launched successfully!")
                    
                    desktop_root.mainloop()
                    
                except Exception as e:
                    error_msg = f"Failed to launch desktop interface: {str(e)}"
                    # Try to log, but if it fails, just print
                    try:
                        self.log_message(f"❌ {error_msg}", "desktop")
                    except:
                        print(f"❌ {error_msg}")
                    messagebox.showerror("Desktop Launch Error", f"{error_msg}\n\nTry restarting the launcher.")
            
            # Schedule the desktop app start after a brief delay
            self.root.after(1000, start_desktop)
            
        except Exception as e:
            error_msg = f"Failed to import desktop interface: {str(e)}"
            self.log_message(f"❌ {error_msg}", "desktop")
            messagebox.showerror("Import Error", error_msg)
    
    def show_loading_window(self, message):
        """Show a loading window."""
        loading = tk.Toplevel(self.root)
        loading.title("Loading...")
        loading.geometry("400x150")
        loading.configure(bg='#0f0f23')
        loading.resizable(False, False)
        
        # Center the loading window
        loading.transient(self.root)
        loading.grab_set()
        
        # Loading content
        loading_frame = tk.Frame(loading, bg='#0f0f23')
        loading_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        loading_label = tk.Label(
            loading_frame,
            text=message,
            font=('Arial', 12),
            fg='#ffffff',
            bg='#0f0f23'
        )
        loading_label.pack(pady=(20, 10))
        
        # Progress bar
        progress = ttk.Progressbar(
            loading_frame,
            mode='indeterminate'
        )
        progress.pack(fill=tk.X, pady=10)
        progress.start()
        
        loading.update()
        return loading
    
    def on_closing(self):
        """Handle application closing."""
        if self.streamlit_process:
            try:
                self.streamlit_process.terminate()
            except:
                pass
        
        self.root.destroy()
    
    def run(self):
        """Run the launcher."""
        self.log_message("🎯 DART Launcher initialized", "system")
        self.log_message("📊 Monitoring system ready", "system")
        self.log_message("🔗 Choose an interface to start", "system")
        self.root.mainloop()
    
    def setup_logs_tab(self):
        """Setup the logs tab with output monitoring."""
        logs_frame = tk.Frame(self.logs_tab, bg='#1a1a3e')
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            logs_frame,
            text="📊 DART Output Logs & Monitoring",
            font=('Arial', 16, 'bold'),
            fg='#38bdf8',
            bg='#1a1a3e'
        )
        title_label.pack(pady=(0, 20))
        
        # Create tabs for different log types
        logs_notebook = ttk.Notebook(logs_frame)
        logs_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Web Dashboard Logs
        self.web_logs_frame = tk.Frame(logs_notebook, bg='#1a1a3e')
        logs_notebook.add(self.web_logs_frame, text="🌐 Web Dashboard")
        
        self.web_logs_text = scrolledtext.ScrolledText(
            self.web_logs_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            bg='#0f0f23',
            fg='#10b981',
            font=('Consolas', 10),
            insertbackground='#38bdf8'
        )
        self.web_logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Desktop Interface Logs
        self.desktop_logs_frame = tk.Frame(logs_notebook, bg='#1a1a3e')
        logs_notebook.add(self.desktop_logs_frame, text="🖥️ Desktop Interface")
        
        self.desktop_logs_text = scrolledtext.ScrolledText(
            self.desktop_logs_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            bg='#0f0f23',
            fg='#f59e0b',
            font=('Consolas', 10),
            insertbackground='#38bdf8'
        )
        self.desktop_logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # System Logs
        self.system_logs_frame = tk.Frame(logs_notebook, bg='#1a1a3e')
        logs_notebook.add(self.system_logs_frame, text="⚙️ System")
        
        self.system_logs_text = scrolledtext.ScrolledText(
            self.system_logs_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            bg='#0f0f23',
            fg='#8b5cf6',
            font=('Consolas', 10),
            insertbackground='#38bdf8'
        )
        self.system_logs_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control buttons
        control_frame = tk.Frame(logs_frame, bg='#1a1a3e')
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        clear_logs_btn = tk.Button(
            control_frame,
            text="🗑️ Clear Logs",
            command=self.clear_logs,
            bg='#ef4444',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=5
        )
        clear_logs_btn.pack(side=tk.LEFT, padx=5)
        
        save_logs_btn = tk.Button(
            control_frame,
            text="💾 Save Logs",
            command=self.save_logs,
            bg='#10b981',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=5
        )
        save_logs_btn.pack(side=tk.LEFT, padx=5)
        
        # Auto-scroll toggle
        self.auto_scroll_var = tk.BooleanVar(value=True)
        auto_scroll_check = tk.Checkbutton(
            control_frame,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            bg='#1a1a3e',
            fg='white',
            selectcolor='#0f0f23',
            font=('Arial', 10)
        )
        auto_scroll_check.pack(side=tk.RIGHT, padx=5)
    
    def log_message(self, message, log_type="system"):
        """Add a message to the appropriate log."""
        try:
            # Check if the root window still exists
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return
                
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            
            # Choose the appropriate log widget
            if log_type == "web":
                log_widget = getattr(self, 'web_logs_text', None)
            elif log_type == "desktop":
                log_widget = getattr(self, 'desktop_logs_text', None)
            else:
                log_widget = getattr(self, 'system_logs_text', None)
            
            # Check if the log widget exists and is valid
            if not log_widget or not log_widget.winfo_exists():
                return
            
            # Add message to log
            log_widget.insert(tk.END, formatted_message)
            
            # Auto-scroll if enabled
            if hasattr(self, 'auto_scroll_var') and self.auto_scroll_var.get():
                log_widget.see(tk.END)
            
            # Limit log size to prevent memory issues
            lines = log_widget.get(1.0, tk.END).split('\n')
            if len(lines) > 1000:
                log_widget.delete(1.0, f"{len(lines)-1000}.0")
            
            # Update UI
            self.root.update_idletasks()
            
        except (tk.TclError, AttributeError):
            # Widget has been destroyed or doesn't exist, silently ignore
            pass
    
    def clear_logs(self):
        """Clear all log contents."""
        try:
            if hasattr(self, 'web_logs_text') and self.web_logs_text.winfo_exists():
                self.web_logs_text.delete(1.0, tk.END)
            if hasattr(self, 'desktop_logs_text') and self.desktop_logs_text.winfo_exists():
                self.desktop_logs_text.delete(1.0, tk.END)
            if hasattr(self, 'system_logs_text') and self.system_logs_text.winfo_exists():
                self.system_logs_text.delete(1.0, tk.END)
            self.log_message("Logs cleared", "system")
        except (tk.TclError, AttributeError):
            pass
    
    def save_logs(self):
        """Save logs to file."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"dart_logs_{timestamp}.txt"
            
            with open(log_file, 'w') as f:
                f.write("=== DART System Logs ===\n")
                if hasattr(self, 'system_logs_text') and self.system_logs_text.winfo_exists():
                    f.write(self.system_logs_text.get(1.0, tk.END))
                f.write("\n=== Web Dashboard Logs ===\n")
                if hasattr(self, 'web_logs_text') and self.web_logs_text.winfo_exists():
                    f.write(self.web_logs_text.get(1.0, tk.END))
                f.write("\n=== Desktop Interface Logs ===\n")
                if hasattr(self, 'desktop_logs_text') and self.desktop_logs_text.winfo_exists():
                    f.write(self.desktop_logs_text.get(1.0, tk.END))
            
            self.log_message(f"Logs saved to {log_file}", "system")
            messagebox.showinfo("Logs Saved", f"Logs saved to {log_file}")
        except Exception as e:
            try:
                self.log_message(f"Error saving logs: {e}", "system")
            except:
                print(f"Error saving logs: {e}")
            messagebox.showerror("Error", f"Failed to save logs: {e}")


if __name__ == "__main__":
    launcher = DARTLauncher()
    launcher.run()
