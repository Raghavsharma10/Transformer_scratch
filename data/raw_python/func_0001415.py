def format_page(text):
    """Format the text for output adding ASCII frame around the text.

    Args:
        text (str): Text that needs to be formatted.

    Returns:
        str: Formatted string.
    """
    width = max(map(len, text.splitlines()))
    page = "+-" + "-" * width + "-+\n"
    for line in text.splitlines():
        page += "| " + line.ljust(width) + " |\n"
    page += "+-" + "-" * width + "-+\n"
    return page