"""
DART - Deep Adaptive Reinforcement Trader
Main entry point for the application.

Choose between modern web dashboard or classic desktop interface.
"""

from dart_launcher_new import DARTLauncher


def main():
    """Launch the DART interface selector."""
    launcher = DARTLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
