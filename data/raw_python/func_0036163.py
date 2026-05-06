def _split_one_end(path):
        """
        Utility function for splitting off the very end part of a path.
        """
        s = path.rsplit('/', 1)
        if len(s) == 1:
            return s[0], ''
        else:
            return tuple(s)