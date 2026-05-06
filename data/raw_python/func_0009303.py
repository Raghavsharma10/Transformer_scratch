def as_uri_2_3(self):
        """
        Returns the value of component encoded as URI string.

        Scans an input string s and applies the following transformations:

        - Pass alphanumeric characters thru untouched
        - Percent-encode quoted non-alphanumerics as needed
        - Unquoted special characters are mapped to their special forms.

        :returns: URI string associated with component
        :rtype: string
        """

        s = self._standard_value
        result = []
        idx = 0
        while (idx < len(s)):
            thischar = s[idx]  # get the idx'th character of s

            # alphanumerics (incl. underscore) pass untouched
            if (CPEComponentSimple._is_alphanum(thischar)):
                result.append(thischar)
                idx += 1
                continue

            # escape character
            if (thischar == "\\"):
                idx += 1
                nxtchar = s[idx]
                result.append(CPEComponentSimple._pct_encode_uri(nxtchar))
                idx += 1
                continue

            # Bind the unquoted '?' special character to "%01".
            if (thischar == "?"):
                result.append("%01")

            # Bind the unquoted '*' special character to "%02".
            if (thischar == "*"):
                result.append("%02")

            idx += 1

        return "".join(result)