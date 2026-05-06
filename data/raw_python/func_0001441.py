def version(self, version_str: str) -> None:
        """
        param version

        Version version number property. Must be a string consisting of three
        non-negative integers delimited by periods (eg. '1.0.1').
        """
        ver = []
        for i in version_str.split('.'):
            ver.append(int(i))
            self.filter_negatives(int(i))
        self._major, self._minor, self._patch = ver[0], ver[1], ver[2]