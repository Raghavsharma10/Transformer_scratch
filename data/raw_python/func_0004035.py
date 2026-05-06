def is_equivalent(self, other):
        """
        Return ``True`` if the IPA character is equivalent to the ``other`` object.

        The ``other`` object can be:

        1. a Unicode string, containing the representation of the IPA character,
        2. a Unicode string, containing a space-separated list of descriptors,
        3. a list of Unicode strings, containing descriptors, and
        4. another IPAChar.

        :rtype: bool
        """
        if (self.unicode_repr is not None) and (is_unicode_string(other)) and (self.unicode_repr == other):
            return True
        if isinstance(other, IPAChar):
            return self.canonical_representation == other.canonical_representation
        try:
            return self.canonical_representation == IPAChar(name=None, descriptors=other).canonical_representation
        except:
            return False