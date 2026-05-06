def version(self) -> str:
        """
        Version version number property. Must be a string consisting of three
        non-negative integers delimited by periods (eg. '1.0.1').
        """
        version: str = (
            str(self._major) + '.' +
            str(self._minor) + '.' +
            str(self._patch)
        )
        return version