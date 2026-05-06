def from_locale(cls, locale):
        """
        Create a new Language instance from a locale string
        :param locale: locale as string
        :return: Language instance with instance.locale() == locale if locale is valid else instance of Unknown Language
        """
        locale = str(locale)
        if locale is 'unknown':
            return UnknownLanguage(locale)
        try:
            return cls._from_xyz('locale', locale)
        except NotALanguageException:
            log.warning('Unknown locale: {}'.format(locale))
            return UnknownLanguage(locale)