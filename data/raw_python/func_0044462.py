def _load_config(self, path):
        """
        Load configuration from JSON

        :param path: path to the JSON config file
        :type path: str
        :return: config dictionary
        :rtype: dict
        """
        p = os.path.abspath(os.path.expanduser(path))
        logger.debug('Loading configuration from: %s', p)
        return read_json_file(p)