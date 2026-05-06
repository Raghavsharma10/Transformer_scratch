def install_locale(cls, locale_code, locale_type):
        """Install the locale specified by `language_code`, for localizations of type `locale_type`.

        If we can't perform localized formatting for the specified locale,
        then the default localization format will be used.

        If the locale specified is already installed for the selected type, then this is a no-op.
        """

        # Skip if the locale is already installed
        if locale_code == getattr(cls, locale_type):
            return
        try:
            # We create a Locale instance to see if the locale code is supported
            locale = Locale(locale_code)
            log.debug('Installed locale %s', locale_code)
        except UnknownLocaleError:
            default = settings.DEFAULT_LOCALIZATION_FORMAT
            log.warning('Unknown locale %s, falling back to %s', locale_code, default)
            locale = Locale(default)
        setattr(cls, locale_type, locale.language)