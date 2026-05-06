def is_equivalent(self, other, ignore=False):
        """
        Return ``True`` if the IPA string is equivalent to the ``other`` object.

        The ``other`` object can be:

        1. a Unicode string,
        2. a list of IPAChar objects, and
        3. another IPAString.

        :param variant other: the object to be compared against
        :param bool ignore: if other is a Unicode string, ignore Unicode characters not IPA valid
        :rtype: bool
        """
        def is_equivalent_to_list_of_ipachars(other):
            """
            Return ``True`` if the list of IPAChar objects
            in the canonical representation of the string
            is the same as the given list.

            :param list other: list of IPAChar objects
            :rtype: bool
            """
            my_ipa_chars = self.canonical_representation.ipa_chars
            if len(my_ipa_chars) != len(other):
                return False
            for i in range(len(my_ipa_chars)):
                if not my_ipa_chars[i].is_equivalent(other[i]):
                    return False
            return True

        if is_unicode_string(other):
            try:
                return is_equivalent_to_list_of_ipachars(IPAString(unicode_string=other, ignore=ignore).ipa_chars)
            except:
                return False
        if is_list_of_ipachars(other):
            try:
                return is_equivalent_to_list_of_ipachars(other) 
            except:
                return False
        if isinstance(other, IPAString):
            return is_equivalent_to_list_of_ipachars(other.canonical_representation.ipa_chars)
        return False