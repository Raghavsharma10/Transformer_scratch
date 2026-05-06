def getCityUsers(self, numberOfThreads=20):
        """Get all the users from the city.

        :param numberOfThreads: number of threads to run.
        :type numberOfThreads: int.
        """
        if not self.__intervals:
            self.__logger.debug("Calculating best intervals")
            self.calculateBestIntervals()

        self.__end = False
        self.__threads = set()

        comprobationURL = self.__getURL()
        self.__readAPI(comprobationURL)

        self.__launchThreads(numberOfThreads)
        self.__logger.debug("Launching threads")
        for i in self.__intervals:
            self.__getPeriodUsers(i[0], i[1])

        self.__end = True

        for t in self.__threads:
            t.join()
        self.__logger.debug("Threads joined")