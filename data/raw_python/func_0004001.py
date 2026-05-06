def map_unicode_string(self, unicode_string, ignore=False, single_char_parsing=False, return_as_list=False, return_can_map=False):
        """
        Convert the given Unicode string, representing an IPA string,
        to a string containing the corresponding mapped representation.

        Return ``None`` if ``unicode_string`` is ``None``.

        :param str unicode_string: the Unicode string to be parsed
        :param bool ignore: if ``True``, ignore Unicode characters that are not IPA valid
        :param bool single_char_parsing: if ``True``, parse one Unicode character at a time
        :param bool return_as_list: if ``True``, return as a list of strings, one for each IPAChar,
                                    instead of their concatenation (single str)
        :param bool return_can_map: if ``True``, return a pair ``(bool, str)``, where the first element
                                    says if the mapper can map all the IPA characters in the given IPA string,
                                    and the second element is either ``None`` or the mapped string/list
        :rtype: str or (bool, str) or (bool, list)
        """
        if unicode_string is None:
            return None
        ipa_string = IPAString(unicode_string=unicode_string, ignore=ignore, single_char_parsing=single_char_parsing)
        return self.map_ipa_string(
            ipa_string=ipa_string,
            ignore=ignore,
            return_as_list=return_as_list,
            return_can_map=return_can_map
        )