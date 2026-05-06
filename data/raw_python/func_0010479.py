def _from_xyz(cls, xyzkey, xyzvalue):
        """
        Private helper function to create new Language instance.
        :param xyzkey: one of ('locale', 'ISO639', 'LanguageID', 'LanguageName')
        :param xyzvalue: corresponding value of xyzkey
        :return: Language instance
        """
        if xyzvalue == 'unknown' or xyzvalue == _('unknown'):
            return UnknownLanguage(xyzvalue)
        for lang_id, lang_data in enumerate(LANGUAGES):
            for data_value in lang_data[xyzkey]:
                if xyzvalue == data_value.lower():
                    return cls(lang_id)
        raise NotALanguageException(xyzvalue, 'Illegal language {}: {}'.format(xyzkey, xyzvalue))