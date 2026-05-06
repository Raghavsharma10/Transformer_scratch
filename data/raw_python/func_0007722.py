def _group(self, value: str):
        """
        Takes a string and groups it appropriately with any
        period or other appropriate punctuation so that it is
        displayed correctly.
        :param value: a string containing an integer or float
        :return: None
        """
        reversed_v = value[::-1]

        parts = []

        has_period = False
        for c in reversed_v:
            if has_period:
                parts.append(c + '.')
                has_period = False
            elif c == '.':
                has_period = True
            else:
                parts.append(c)

        parts = parts[:len(self._digits)]

        return parts