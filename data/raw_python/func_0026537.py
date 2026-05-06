def i18n(msg, event=None, lang='en', domain='backend'):
    """Gettext function wrapper to return a message in a specified language by domain

    To use internationalization (i18n) on your messages, import it as '_' and use as usual.
    Do not forget to supply the client's language setting."""

    if event is not None:
        language = event.client.language
    else:
        language = lang

    domain = Domain(domain)
    return domain.get(language, msg)