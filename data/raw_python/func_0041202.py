def translation(language):
    """
    Return a translation object in the default 'django' domain.
    """
    global _translations
    if language not in _translations:
        _translations[language] = Translations(language)
    return _translations[language]