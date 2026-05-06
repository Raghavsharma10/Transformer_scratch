def get_relative_from_paths(self, filepath, paths):
        """
        Find the relative filepath from the most relevant multiple paths.

        This is somewhat like a ``os.path.relpath(path[, start])`` but where
        ``start`` is a list. The most relevant item from ``paths`` will be used
        to apply the relative transform.

        Args:
            filepath (str): Path to transform to relative.
            paths (list): List of absolute paths to use to find and remove the
                start path from ``filepath`` argument. If there is multiple
                path starting with the same directories, the biggest will
                match.

        Raises:
            boussole.exception.FinderException: If no ``filepath`` start could
            be finded.

        Returns:
            str: Relative filepath where the start coming from ``paths`` is
                removed.
        """
        for systempath in paths_by_depth(paths):
            if filepath.startswith(systempath):
                return os.path.relpath(filepath, systempath)

        raise FinderException("'Finder.get_relative_from_paths()' could not "
                              "find filepath start from '{}'".format(filepath))