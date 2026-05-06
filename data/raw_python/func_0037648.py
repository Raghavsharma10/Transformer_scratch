def _get_next_occurrence(haystack, offset, needles):
        """
        Find next occurence of one of the needles in the haystack

        :return: tuple of (index, needle found)
             or: None if no needle was found"""
        # make map of first char to full needle (only works if all needles
        # have different first characters)
        firstcharmap = dict([(n[0], n) for n in needles])
        firstchars = firstcharmap.keys()
        while offset < len(haystack):
            if haystack[offset] in firstchars:
                possible_needle = firstcharmap[haystack[offset]]
                if haystack[offset:offset + len(possible_needle)] == possible_needle:
                    return offset, possible_needle
            offset += 1
        return None