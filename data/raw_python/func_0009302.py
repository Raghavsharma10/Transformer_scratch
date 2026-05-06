def as_fs(self):
        """
        Returns the value of component encoded as formatted string.

        Inspect each character in value of component.
        Certain nonalpha characters pass thru without escaping
        into the result, but most retain escaping.

        :returns: Formatted string associated with component
        :rtype: string
        """

        s = self._standard_value
        result = []
        idx = 0
        while (idx < len(s)):

            c = s[idx]  # get the idx'th character of s
            if c != "\\":
                # unquoted characters pass thru unharmed
                result.append(c)
            else:
                # Escaped characters are examined
                nextchr = s[idx + 1]

                if (nextchr == ".") or (nextchr == "-") or (nextchr == "_"):
                    # the period, hyphen and underscore pass unharmed
                    result.append(nextchr)
                    idx += 1
                else:
                    # all others retain escaping
                    result.append("\\")
                    result.append(nextchr)
                    idx += 2
                    continue
            idx += 1

        return "".join(result)