def isubset(self, *keys):
        # type: (*Hashable) -> ww.g
        """Return key, self[key] as generator for key in keys.

        Raise KeyError if a key does not exist

        Args:
            keys: Iterable containing keys

        Example:

            >>> from ww import d
            >>> list(d({1: 1, 2: 2, 3: 3}).isubset(1, 3))
            [(1, 1), (3, 3)]
        """
        return ww.g((key, self[key]) for key in keys)