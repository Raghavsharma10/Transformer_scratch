def render_text(text, language=None):
    """
    Render the text, reuses the template filters provided by Django.
    """
    # Get the filter
    text_filter = SUPPORTED_LANGUAGES.get(language, None)
    if not text_filter:
        raise ImproperlyConfigured("markup filter does not exist: {0}. Valid options are: {1}".format(
            language, ', '.join(list(SUPPORTED_LANGUAGES.keys()))
        ))

    # Convert.
    return text_filter(text)