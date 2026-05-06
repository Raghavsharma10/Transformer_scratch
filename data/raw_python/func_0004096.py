def latinize_text(text, ascii=False):
    """Transliterate the given text to the latin script.

    This attempts to convert a given text to latin script using the
    closest match of characters vis a vis the original script.
    """
    if text is None or not isinstance(text, six.string_types) or not len(text):
        return text

    if ascii:
        if not hasattr(latinize_text, '_ascii'):
            # Transform to latin, separate accents, decompose, remove
            # symbols, compose, push to ASCII
            latinize_text._ascii = Transliterator.createInstance('Any-Latin; NFKD; [:Symbol:] Remove; [:Nonspacing Mark:] Remove; NFKC; Accents-Any; Latin-ASCII')  # noqa
        return latinize_text._ascii.transliterate(text)

    if not hasattr(latinize_text, '_tr'):
        latinize_text._tr = Transliterator.createInstance('Any-Latin')
    return latinize_text._tr.transliterate(text)