def parents(self, sourcepath, recursive=True):
        """
        Recursively find all parents that import the given source path.

        Args:
            sourcepath (str): Source file path to search for.

        Keyword Arguments:
            recursive (bool): Switch to enabled recursive finding (if True).
                Default to True.

        Returns:
            set: List of finded parents path.
        """
        return self._get_recursive_dependancies(
            self._PARENTS_MAP,
            sourcepath,
            recursive=True
        )