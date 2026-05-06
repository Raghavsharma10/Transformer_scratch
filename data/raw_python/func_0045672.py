def find_next(self, *strings, **kwargs):
        """
        From the editor's current cursor position find the next instance of the
        given string.

        Args:
            strings (iterable): String or strings to search for

        Returns:
            tup (tuple): Tuple of cursor position and line or None if not found

        Note:
            This function cycles the entire editor (i.e. cursor to length of
            editor to zero and back to cursor position).
        """
        start = kwargs.pop("start", None)
        keys_only = kwargs.pop("keys_only", False)
        staht = start if start is not None else self.cursor
        for start, stop in [(staht, len(self)), (0, staht)]:
            for i in range(start, stop):
                for string in strings:
                    if string in self[i]:
                        tup = (i, self[i])
                        self.cursor = i + 1
                        if keys_only: return i
                        return tup