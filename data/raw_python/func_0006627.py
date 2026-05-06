def lookup_by_name(self, name):
        """
        Function for retrieving the UnicodeCharacter associated with a name.  The name lookup uses the loose matching
        rule UAX44-LM2 for loose matching.  See the following for more info:

        https://www.unicode.org/reports/tr44/#UAX44-LM2

        For example:

        ucd = UnicodeData()
        ucd.lookup_by_name("LATIN SMALL LETTER SHARP S") -> UnicodeCharacter(name='LATIN SMALL LETTER SHARP S',...)
        ucd.lookup_by_name("latin_small_letter_sharp_s") -> UnicodeCharacter(name='LATIN SMALL LETTER SHARP S',...)


        :param name: Name of the character to look up.
        :return: UnicodeCharacter instance with data associated with the character.
        """
        try:
            return self._name_database[_uax44lm2transform(name)]
        except KeyError:
            raise KeyError(u"Unknown character name: '{0}'!".format(name))