def _tofloat(obj):
    """Convert to float if object is a float string."""
    if "inf" in obj.lower().strip():
        return obj
    try:
        return int(obj)
    except ValueError:
        try:
            return float(obj)
        except ValueError:
            return obj