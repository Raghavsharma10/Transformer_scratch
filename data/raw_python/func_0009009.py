def walkdirs(self, pattern=None, errors='strict'):
        """ D.walkdirs() -> iterator over subdirs, recursively.

        With the optional `pattern` argument, this yields only
        directories whose names match the given pattern.  For
        example, ``mydir.walkdirs('*test')`` yields only directories
        with names ending in ``'test'``.

        The `errors=` keyword argument controls behavior when an
        error occurs.  The default is ``'strict'``, which causes an
        exception.  The other allowed values are ``'warn'`` (which
        reports the error via :func:`warnings.warn()`), and ``'ignore'``.
        """
        if errors not in ('strict', 'warn', 'ignore'):
            raise ValueError("invalid errors parameter")

        try:
            dirs = self.dirs()
        except Exception:
            if errors == 'ignore':
                return
            elif errors == 'warn':
                warnings.warn(
                    "Unable to list directory '%s': %s"
                    % (self, sys.exc_info()[1]),
                    TreeWalkWarning)
                return
            else:
                raise

        for child in dirs:
            if pattern is None or child.fnmatch(pattern):
                yield child
            for subsubdir in child.walkdirs(pattern, errors):
                yield subsubdir