def color_string(color, string):
    """
    Colorizes a given string, if coloring is available.
    """
    if not color_available:
        return string

    return color + string + colorama.Fore.RESET