def center_text(text, width=80):
    """Center all lines of the text.

    It is assumed that all lines width is smaller then B{width}, because the
    line width will not be checked.

    Args:
        text (str): Text to wrap.
        width (int): Maximum number of characters per line.

    Returns:
        str: Centered text.
    """
    centered = []
    for line in text.splitlines():
        centered.append(line.center(width))
    return "\n".join(centered)