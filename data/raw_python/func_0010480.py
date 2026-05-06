def from_unknown(cls, value, xx=False, xxx=False, locale=False, name=False):
        """
        Try to create a Language instance having only some limited data about the Language.
        If no corresponding Language is found, a NotALanguageException is thrown.
        :param value: data known about the language as string
        :param xx: True if the value may be a locale
        :param xxx: True if the value may be a LanguageID
        :param locale: True if the value may be a locale
        :param name: True if the value may be a LanguageName
        :return: Language Instance if a matching Language was found
        """
        # Use 2 lists instead of dict ==> order known
        keys = ['ISO639', 'LanguageID', 'locale', 'LanguageName']
        truefalses = [xx, xxx, locale, name]
        value = value.lower()
        for key, doKey in zip(keys, truefalses):
            if doKey:
                try:
                    return cls._from_xyz(key, value)
                except NotALanguageException:
                    pass
        raise NotALanguageException(value, 'Illegal language "{}"'.format(value))