def coerce_to_int(val, default=0xDEADBEEF):
    """Attempts to cast given value to an integer, return the original value if failed or the default if one provided."""
    try:
        return int(val)
    except (TypeError, ValueError):
        if default != 0xDEADBEEF:
            return default
        return val