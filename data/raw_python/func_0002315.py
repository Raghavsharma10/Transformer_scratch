def softhyphen_filter(textitem, html):
    """
    Apply soft hyphenation to the text, which inserts ``&shy;`` markers.
    """
    language = textitem.language_code

    # Make sure the Django language code gets converted to what django-softhypen 1.0.2 needs.
    if language == 'en':
        language = 'en-us'
    elif '-' not in language:
        language = "{0}-{0}".format(language)

    return hyphenate(html, language=language)