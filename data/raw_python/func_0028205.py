def append(self, *values):
        """Append values at the end of the list

        Allow chaining.

        Args:
            values: values to be appened at the end.

        Example:

            >>> from ww import l
            >>> lst = l([])
            >>> lst.append(1)
            [1]
            >>> lst
            [1]
            >>> lst.append(2, 3).append(4,5)
            [1, 2, 3, 4, 5]
            >>> lst
            [1, 2, 3, 4, 5]
        """

        for value in values:
            list.append(self, value)
        return self