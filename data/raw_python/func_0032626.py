def getDocFactory(self, fragmentName, default=None):
        """
        Retrieve a Nevow document factory for the given name.

        @param fragmentName: a short string that names a fragment template.

        @param default: value to be returned if the named template is not
        found.
        """
        themes = self._preferredThemes()
        for t in themes:
            fact = t.getDocFactory(fragmentName, None)
            if fact is not None:
                return fact
        return default