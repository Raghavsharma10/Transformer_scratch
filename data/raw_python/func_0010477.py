def from_xxx(cls, xxx):
        """
        Create a new Language instance from a LanguageID string
        :param xxx: LanguageID as string
        :return: Language instance with instance.xxx() == xxx if xxx is valid else instance of UnknownLanguage
        """
        xxx = str(xxx).lower()
        if xxx is 'unknown':
            return UnknownLanguage(xxx)
        try:
            return cls._from_xyz('LanguageID', xxx)
        except NotALanguageException:
            log.warning('Unknown LanguageId: {}'.format(xxx))
            return UnknownLanguage(xxx)