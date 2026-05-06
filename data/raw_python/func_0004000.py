def map_ipa_string(self, ipa_string, ignore=False, return_as_list=False, return_can_map=False):
        """
        Convert the given IPAString to a string
        containing the corresponding ASCII IPA representation.

        :param IPAString ipa_string: the IPAString to be parsed
        :param bool ignore: if ``True``, ignore Unicode characters that are not IPA valid
        :param bool return_as_list: if ``True``, return as a list of strings, one for each IPAChar,
                                    instead of their concatenation (single str)
        :param bool return_can_map: if ``True``, return a pair ``(bool, str)``, where the first element
                                    says if the mapper can map all the IPA characters in the given IPA string,
                                    and the second element is either ``None`` or the mapped string/list
        :rtype: str or (bool, str) or (bool, list)
        """
        acc = []
        can_map = True
        canonical = [(c.canonical_representation, ) for c in ipa_string]
        split = split_using_dictionary(canonical, self, self.max_key_length, single_char_parsing=False)
        for sub in split:
            try:
                acc.append(self.ipa_canonical_representation_to_mapped_str[sub])
            except KeyError:
                if ignore:
                    can_map = False
                else:
                    raise ValueError("The IPA string contains an IPA character that is not mapped: %s" % sub)
        mapped = acc if return_as_list else u"".join(acc)
        if return_can_map:
            return (can_map, mapped)
        return mapped