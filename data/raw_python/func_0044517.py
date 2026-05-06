def getConfig(self):
        """Return the configuration of the city.

        :return: configuration of the city.
        :rtype: dict.
        """
        config = {}
        config["name"] = self.city
        config["intervals"] = self.__intervals
        config["last_date"] = self.__lastDay
        config["excludedUsers"] = []
        config["excludedLocations"] = []

        for e in self.__excludedUsers:
            config["excludedUsers"].append(e)

        for e in self.__excludedLocations:
            config["excludedLocations"].append(e)

        config["locations"] = self.__locations
        return config