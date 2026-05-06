def is_listish(obj):
    """Check if something quacks like a list."""
    if isinstance(obj, (list, tuple, set)):
        return True
    return is_sequence(obj)