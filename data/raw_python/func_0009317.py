def _decode(self):
        """
        Convert the characters of character in value of component to standard
        value (WFN value).
        This function scans the value of component and returns a copy
        with all percent-encoded characters decoded.

        :exception: ValueError - invalid character in value of component
        """

        result = []
        idx = 0
        s = self._encoded_value
        embedded = False

        errmsg = []
        errmsg.append("Invalid value: ")

        while (idx < len(s)):
            errmsg.append(s)
            errmsg_str = "".join(errmsg)

            # Get the idx'th character of s
            c = s[idx]

            # Deal with dot, hyphen and tilde: decode with quoting
            if ((c == '.') or (c == '-') or (c == '~')):
                result.append("\\")
                result.append(c)
                idx += 1
                embedded = True  # a non-%01 encountered
                continue

            if (c != '%'):
                result.append(c)
                idx += 1
                embedded = True  # a non-%01 encountered
                continue

            # we get here if we have a substring starting w/ '%'
            form = s[idx: idx + 3]  # get the three-char sequence

            if form == CPEComponent2_3_URI.WILDCARD_ONE:
                # If %01 legal at beginning or end
                # embedded is false, so must be preceded by %01
                # embedded is true, so must be followed by %01
                if (((idx == 0) or (idx == (len(s)-3))) or
                    ((not embedded) and (s[idx - 3:idx] == CPEComponent2_3_URI.WILDCARD_ONE)) or
                    (embedded and (len(s) >= idx + 6) and (s[idx + 3:idx + 6] == CPEComponent2_3_URI.WILDCARD_ONE))):

                    # A percent-encoded question mark is found
                    # at the beginning or the end of the string,
                    # or embedded in sequence as required.
                    # Decode to unquoted form.
                    result.append(CPEComponent2_3_WFN.WILDCARD_ONE)
                    idx += 3
                    continue
                else:
                    raise ValueError(errmsg_str)

            elif form == CPEComponent2_3_URI.WILDCARD_MULTI:
                if ((idx == 0) or (idx == (len(s) - 3))):
                    # Percent-encoded asterisk is at the beginning
                    # or the end of the string, as required.
                    # Decode to unquoted form.
                    result.append(CPEComponent2_3_WFN.WILDCARD_MULTI)
                else:
                    raise ValueError(errmsg_str)

            elif form in CPEComponent2_3_URI.pce_char_to_decode.keys():
                value = CPEComponent2_3_URI.pce_char_to_decode[form]
                result.append(value)

            else:
                errmsg.append("Invalid percent-encoded character: ")
                errmsg.append(s)
                raise ValueError("".join(errmsg))

            idx += 3
            embedded = True  # a non-%01 encountered.

        self._standard_value = "".join(result)