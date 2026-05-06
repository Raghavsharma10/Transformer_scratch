def is_allowed(self, filepath, excludes=[]):
        """
        Check from exclude patterns if a relative filepath is allowed

        Args:
            filepath (str): A relative file path. (exclude patterns are
                allways based from the source directory).

        Keyword Arguments:
            excludes (list): A list of excluding (glob) patterns. If filepath
                matchs one of patterns, filepath is not allowed.

        Raises:
            boussole.exception.FinderException: If given filepath is absolute.

        Returns:
            str: Filepath with new extension.
        """
        if os.path.isabs(filepath):
            raise FinderException("'Finder.is_allowed()' only accept relative"
                                  " filepath")

        if excludes:
            for pattern in excludes:
                if fnmatch.fnmatch(filepath, pattern):
                    return False
        return True