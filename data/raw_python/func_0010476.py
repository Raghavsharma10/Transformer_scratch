def from_xx(cls, xx):
        """
        Create a new Language instance from a ISO639 string
        :param xx: ISO639 as string
        :return: Language instance with instance.xx() == xx if xx is valid else instance of UnknownLanguage
        """
        xx = str(xx).lower()
        if xx is 'unknown':
            return UnknownLanguage(xx)
        try:
            return cls._from_xyz('ISO639', xx)
        except NotALanguageException:
            log.warning('Unknown ISO639: {}'.format(xx))
            return UnknownLanguage(xx)