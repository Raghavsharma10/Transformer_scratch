def _decode(self):
        """
        Convert the characters of string s to standard value (WFN value).
        Inspect each character in value of component. Copy quoted characters,
        with their escaping, into the result. Look for unquoted non
        alphanumerics and if not "*" or "?", add escaping.

        :exception: ValueError - invalid character in value of component
        """

        result = []
        idx = 0
        s = self._encoded_value
        embedded = False

        errmsg = []
        errmsg.append("Invalid character '")

        while (idx < len(s)):
            c = s[idx]  # get the idx'th character of s
            errmsg.append(c)
            errmsg.append("'")
            errmsg_str = "".join(errmsg)

            if (CPEComponentSimple._is_alphanum(c)):
                # Alphanumeric characters pass untouched
                result.append(c)
                idx += 1
                embedded = True
                continue

            if c == "\\":
                # Anything quoted in the bound string stays quoted
                # in the unbound string.
                result.append(s[idx: idx + 2])
                idx += 2
                embedded = True
                continue

            if (c == CPEComponent2_3_FS.WILDCARD_MULTI):
                # An unquoted asterisk must appear at the beginning or
                # end of the string.
                if (idx == 0) or (idx == (len(s) - 1)):
                    result.append(c)
                    idx += 1
                    embedded = True
                    continue
                else:
                    raise ValueError(errmsg_str)

            if (c == CPEComponent2_3_FS.WILDCARD_ONE):
                # An unquoted question mark must appear at the beginning or
                # end of the string, or in a leading or trailing sequence:
                # - ? legal at beginning or end
                # - embedded is false, so must be preceded by ?
                # - embedded is true, so must be followed by ?
                if (((idx == 0) or (idx == (len(s) - 1))) or
                   ((not embedded) and (s[idx - 1] == CPEComponent2_3_FS.WILDCARD_ONE)) or
                   (embedded and (s[idx + 1] == CPEComponent2_3_FS.WILDCARD_ONE))):
                    result.append(c)
                    idx += 1
                    embedded = False
                    continue
                else:
                    raise ValueError(errmsg_str)

            # all other characters must be quoted
            result.append("\\")
            result.append(c)
            idx += 1
            embedded = True

        self._standard_value = "".join(result)