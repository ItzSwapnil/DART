def get_granularity_mapping():
    """Return a mapping of timeframe strings to granularity values in seconds."""
    return {
        "1 minute": 60,
        "2 minutes": 120,
        "3 minutes": 180,
        "5 minutes": 300,
        "10 minutes": 600,
        "15 minutes": 900,
        "30 minutes": 1800,
        "1 hour": 3600,
        "2 hours": 7200,
        "4 hours": 14400,
        "8 hours": 28800,
        "1 day": 86400,
    }
