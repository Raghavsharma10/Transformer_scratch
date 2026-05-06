def install_language(cls, language_code):
        """Install the translations for language specified by `language_code`.

        If we don't have translations for this language, then the default language will be used.

        If the language specified is already installed, then this is a no-op.
        """
        # Skip if the language is already installed
        if language_code == cls.language:
            return
        try:
            cls._active_catalogs = cls._translation_catalogs[language_code]
            cls.language = language_code
            log.debug('Installed language %s', language_code)
        except KeyError:
            default = settings.DEFAULT_LANG
            log.warning('Unknown language %s, falling back to %s', language_code, default)
            cls._active_catalogs = cls._translation_catalogs[default]
            cls.language = default