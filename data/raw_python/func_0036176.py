def get_directories_with_extensions(self, start, extensions=None):
        """
        Look for directories with image extensions in given directory and
        return a list with found dirs.

        .. note:: In deep file structures this might get pretty slow.
        """
        return set([p.parent for ext in extensions for p in start.rglob(ext)])