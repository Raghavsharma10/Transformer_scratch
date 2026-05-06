def from_name(cls, name):
        """
        Create a new Language instance from a name as string
        :param name: name as string
        :return: Language instance with instance.name() == name if name is valid else instance of UnknownLanguage
        """
        name = str(name).lower()
        if name is 'unknown' or name is _('unknown'):
            return UnknownLanguage(name)
        try:
            return cls._from_xyz('LanguageName', name)
        except NotALanguageException:
            log.warning('Unknown LanguageName: {}'.format(name))
            return UnknownLanguage(name)