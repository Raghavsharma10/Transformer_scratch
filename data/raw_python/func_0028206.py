def extend(self, *iterables):
        """Add all values of all iterables at the end of the list

        Args:
            iterables: iterable which content to add at the end

        Example:

            >>> from ww import l
            >>> lst = l([])
            >>> lst.extend([1, 2])
            [1, 2]
            >>> lst
            [1, 2]
            >>> lst.extend([3, 4]).extend([5, 6])
            [1, 2, 3, 4, 5, 6]
            >>> lst
            [1, 2, 3, 4, 5, 6]
        """

        for value in iterables:
            list.extend(self, value)
        return self