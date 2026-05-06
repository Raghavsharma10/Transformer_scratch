def canonical_representation(self):
        """
        Return a new IPAString, containing the canonical representation of the current string,
        that is, the one composed by the (prefix) minimum number of IPAChar objects.

        :rtype: IPAString
        """
        return IPAString(unicode_string=u"".join([c.__unicode__() for c in self.ipa_chars]))