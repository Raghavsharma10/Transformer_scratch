def cols_str(columns):
    """Concatenate list of columns into a string."""
    cols = ""
    for c in columns:
        cols = cols + wrap(c) + ', '
    return cols[:-2]