def configToJson(self, fileName):
        """Save the configuration of the city in a JSON.

        :param fileName: path to the output file.
        :type fileName: str.
        """
        config = self.getConfig()
        with open(fileName, "w") as outfile:
            dump(config, outfile, indent=4, sort_keys=True)