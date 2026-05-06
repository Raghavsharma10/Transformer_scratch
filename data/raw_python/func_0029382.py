def hrule(width=None, char=None):
    """Outputs or returns a horizontal line of the given character and width.
    Returns printed string."""
    width = width or HRWIDTH
    char = char or HRCHAR
    return echo(getline(char, width))