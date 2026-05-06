def can_map_ipa_string(self, ipa_string):
        """
        Return ``True`` if the mapper can map all the IPA characters
        in the given IPA string.

        :param IPAString ipa_string: the IPAString to be parsed
        :rtype: bool
        """
        canonical = [(c.canonical_representation, ) for c in ipa_string]
        split = split_using_dictionary(canonical, self, self.max_key_length, single_char_parsing=False)
        for sub in split:
            if not sub in self.ipa_canonical_representation_to_mapped_str:
                return False
        return True