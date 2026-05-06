def readConfig(self, configuration):
        """Read configuration from dict.

        Read configuration from a JSON configuration file.

        :param configuration: configuration to load.
        :type configuration: dict.
        """
        self.__logger.debug("Reading configuration")
        self.city = configuration["name"]
        self.__logger.info("City name: " + self.city)
        if "intervals" in configuration:
            self.__intervals = configuration["intervals"]
            self.__logger.debug("Intervals: " +
                                str(self.__intervals))

        if "last_date" in configuration:
            self.__lastDay = configuration["last_date"]
            self.__logger.debug("Last day: " + self.__lastDay)

        if "locations" in configuration:
            self.__locations = configuration["locations"]
            self.__logger.debug("Locations: " +
                                str(self.__locations))
            self.__addLocationsToURL(self.__locations)

        if "excludedUsers" in configuration:
            self.__excludedUsers= set()
            self.__excludedLocations = set()

            excluded = configuration["excludedUsers"]
            for e in excluded:
                self.__excludedUsers.add(e)
            self.__logger.debug("Excluded users " +
                                str(self.__excludedUsers))

        if "excludedLocations" in configuration:
            excluded = configuration["excludedLocations"]
            for e in excluded:
                self.__excludedLocations.add(e)

            self.__logger.debug("Excluded locations " +
                                str(self.__excludedLocations))