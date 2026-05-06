def category_replace(text, replacements=UNICODE_CATEGORIES):
    """Remove characters from a string based on unicode classes.

    This is a method for removing non-text characters (such as punctuation,
    whitespace, marks and diacritics) from a piece of text by class, rather
    than specifying them individually.
    """
    if text is None:
        return None
    characters = []
    for character in decompose_nfkd(text):
        cat = category(character)
        replacement = replacements.get(cat, character)
        if replacement is not None:
            characters.append(replacement)
    return u''.join(characters)