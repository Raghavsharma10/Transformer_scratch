def rjust_text(text, width=80, indent=0, subsequent=None):
    """Wrap text and adjust it to right border.

    Same as L{wrap_text} with the difference that the text is aligned against
    the right text border.

    Args:
        text (str): Text to wrap and align.
        width (int): Maximum number of characters per line.
        indent (int): Indentation of the first line.
        subsequent (int or None): Indentation of all other lines, if it is
            ``None``, then the indentation will be same as for the first line.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if subsequent is None:
        subsequent = indent
    wrapper = TextWrapper(
        width=width,
        break_long_words=False,
        replace_whitespace=True,
        initial_indent=" " * (indent + subsequent),
        subsequent_indent=" " * subsequent,
    )
    return wrapper.fill(text)[subsequent:]