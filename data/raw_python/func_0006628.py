def lookup_by_partial_name(self, partial_name):
        """
        Similar to lookup_by_name(name), this method uses loose matching rule UAX44-LM2 to attempt to find the
        UnicodeCharacter associated with a name.  However, it attempts to permit even looser matching by doing a
        substring search instead of a simple match.  This method will return a generator that yields instances of
        UnicodeCharacter where the partial_name passed in is a substring of the full name.

        For example:

        >>> ucd = UnicodeData()
        >>> for data in ucd.lookup_by_partial_name("SHARP S"):
        >>>     print(data.code + " " + data.name)
        >>>
        >>> U+00DF LATIN SMALL LETTER SHARP S
        >>> U+1E9E LATIN CAPITAL LETTER SHARP S
        >>> U+266F MUSIC SHARP SIGN

        :param partial_name: Partial name of the character to look up.
        :return: Generator that yields instances of UnicodeCharacter.
        """
        for k, v in self._name_database.items():
            if _uax44lm2transform(partial_name) in k:
                yield v