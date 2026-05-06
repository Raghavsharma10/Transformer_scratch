def get_version(self, as_tuple=False):
        """Returns uWSGI version string or tuple.

        :param bool as_tuple:

        :rtype: str|tuple
        """
        if as_tuple:
            return uwsgi.version_info

        return decode(uwsgi.version)