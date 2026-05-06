def _getConfigPath(self):
        """
        Gets the currently configured config path.
        :return: the path, raises ValueError if it doesn't exist.
        """
        confHome = environ.get('VIBE_CONFIG_HOME')
        return confHome if confHome is not None else path.join(path.expanduser("~"), '.vibe')