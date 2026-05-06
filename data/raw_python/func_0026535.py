def language_token_to_name(languages):
    """Get a descriptive title for all languages"""

    result = {}

    with open(os.path.join(localedir, 'languages.json'), 'r') as f:
        language_lookup = json.load(f)

    for language in languages:
        language = language.lower()
        try:
            result[language] = language_lookup[language]
        except KeyError:
            l10n_log('Language token lookup not found:', language, lvl=warn)
            result[language] = language

    return result