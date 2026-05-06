def _trim(cls, s):
        """
        Remove trailing colons from the URI back to the first non-colon.

        :param string s: input URI string
        :returns: URI string with trailing colons removed
        :rtype: string

        TEST: trailing colons necessary

        >>> s = '1:2::::'
        >>> CPE._trim(s)
        '1:2'

        TEST: trailing colons not necessary

        >>> s = '1:2:3:4:5:6'
        >>> CPE._trim(s)
        '1:2:3:4:5:6'
        """
        reverse = s[::-1]
        idx = 0
        for i in range(0, len(reverse)):
            if reverse[i] == ":":
                idx += 1
            else:
                break

        # Return the substring after all trailing colons,
        # reversed back to its original character order.
        new_s = reverse[idx: len(reverse)]
        return new_s[::-1]