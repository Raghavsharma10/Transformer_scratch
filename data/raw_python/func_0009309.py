def _decode(self):
        """
        Convert the encoded value of component to standard value (WFN value).
        """

        s = self._encoded_value
        elements = s.replace('~', '').split('!')
        dec_elements = []

        for elem in elements:
            result = []
            idx = 0
            while (idx < len(elem)):
                # Get the idx'th character of s
                c = elem[idx]

                if (c in CPEComponent1_1.NON_STANDARD_VALUES):
                    # Escape character
                    result.append("\\")
                    result.append(c)
                else:
                    # Do nothing
                    result.append(c)

                idx += 1
            dec_elements.append("".join(result))

        self._standard_value = dec_elements