def shorten(text):
    """ Reduce text length for displaying / logging purposes.
    """
    if len(text) >= MAX_DISPLAY_LEN:
        text = text[:MAX_DISPLAY_LEN//2]+"..."+text[-MAX_DISPLAY_LEN//2:]
    return text