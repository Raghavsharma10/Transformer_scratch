def get_accent_string(string):
    """
    Get the first accent from the right of a string.
    """
    accents = list(filter(lambda accent: accent != Accent.NONE,
                          map(get_accent_char, string)))
    return accents[-1] if accents else Accent.NONE