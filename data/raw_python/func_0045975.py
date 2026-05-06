def single_encode(input, errors='strict'):
    """Transliterate to 8 bit using only single letter replacements.

    For example, \u2639 WHITE FROWNING FACE ``☹`` will be passed
    through unchanged.

    """
    if not isinstance(input, text_type):
        input = text_type(input, sys.getdefaultencoding(), errors)
    length = len(input)
    input = unicodedata.normalize('NFKC', input)
    return input.translate(single_table), length