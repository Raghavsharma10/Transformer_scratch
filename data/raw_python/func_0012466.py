def _shorten_version(ver, num_components=2):
        """
        If ``ver`` is a dot-separated string with at least (num_components +1)
        components, return only the first two. Else return the original string.

        :param ver: version string
        :type ver: str
        :return: shortened (major, minor) version
        :rtype: str
        """
        parts = ver.split('.')
        if len(parts) <= num_components:
            return ver
        return '.'.join(parts[:num_components])