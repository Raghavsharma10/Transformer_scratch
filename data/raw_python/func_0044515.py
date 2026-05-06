def readConfigFromJSON(self, fileName):
        """Read configuration from JSON.

        :param fileName: path to the configuration file.
        :type fileName: str.
        """
        self.__logger.debug("readConfigFromJSON: reading from " + fileName)
        with open(fileName) as data_file:
            data = load(data_file)
        self.readConfig(data)