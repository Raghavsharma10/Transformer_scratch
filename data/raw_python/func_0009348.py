def _decode(self):
        """
        Convert the encoded value of component to standard value (WFN value).
        """

        result = []
        idx = 0
        s = self._encoded_value

        while (idx < len(s)):
            # Get the idx'th character of s
            c = s[idx]

            if (c in CPEComponent2_2.NON_STANDARD_VALUES):
                # Escape character
                result.append("\\")
                result.append(c)
            else:
                # Do nothing
                result.append(c)

            idx += 1

        self._standard_value = "".join(result)