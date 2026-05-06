def match_file(self, f, update_file=False):
        """
        Determine whether the passed file matches the Entity.

        Args:
            f (File): The File instance to match against.

        Returns: the matched value if a match was found, otherwise None.
        """
        if self.map_func is not None:
            val = self.map_func(f)
        else:
            m = self.regex.search(f.path)
            val = m.group(1) if m is not None else None

        return self._astype(val)