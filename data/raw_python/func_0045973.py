def long_encode(input, errors='strict'):
    """Transliterate to 8 bit using as many letters as needed.

    For example, \u00e4 LATIN SMALL LETTER A WITH DIAERESIS ``ä`` will
    be replaced with ``ae``.

    """
    if not isinstance(input, text_type):
        input = text_type(input, sys.getdefaultencoding(), errors)
    length = len(input)
    input = unicodedata.normalize('NFKC', input)
    return input.translate(long_table), length