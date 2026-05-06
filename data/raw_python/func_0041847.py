def to_language(locale):
    """
    Turns a locale name (en_US) into a language name (en-us).

    Taken `from Django <http://bit.ly/1vWACbE>`_.
    """
    p = locale.find('_')
    if p >= 0:
        return locale[:p].lower() + '-' + locale[p + 1:].lower()
    else:
        return locale.lower()