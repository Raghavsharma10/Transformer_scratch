def drop_bad_characters(text):
    """Takes a text and drops all non-printable and non-ascii characters and
    also any whitespace characters that aren't space.

    :arg str text: the text to fix

    :returns: text with all bad characters dropped

    """
    # Strip all non-ascii and non-printable characters
    text = ''.join([c for c in text if c in ALLOWED_CHARS])
    return text