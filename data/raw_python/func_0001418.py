def wrap_text(text, width=80):
    """Wrap text lines to maximum *width* characters.

    Wrapped text is aligned against the left text border.

    Args:
        text (str): Text to wrap.
        width (int): Maximum number of characters per line.

    Returns:
        str: Wrapped text.
    """
    text = re.sub(r"\s+", " ", text).strip()
    wrapper = TextWrapper(
        width=width, break_long_words=False, replace_whitespace=True
    )
    return wrapper.fill(text)