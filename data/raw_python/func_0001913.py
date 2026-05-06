def _load_yaml_file(self, file):
        """
        Load data from yaml file

        :param file: Readable object or path to file
        :type file: FileIO | str | unicode
        :return: Yaml data
        :rtype: None | int | float | str | unicode | list | dict
        :raises IOError: Failed to load
        """
        try:
            res = load_yaml_file(file)
        except:
            self.exception("Failed to load from {}".format(file))
            raise IOError("Loading file failed")
        return res