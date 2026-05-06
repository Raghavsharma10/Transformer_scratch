def split_elements(value):
    """Split a string with comma or space-separated elements into a list."""
    l = [v.strip() for v in value.split(',')]
    if len(l) == 1:
        l = value.split()
    return l