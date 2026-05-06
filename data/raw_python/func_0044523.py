def calculateBestIntervals(self):
        """Calcule valid intervals of a city."""
        self.__intervals = []
        self.__readAPI(self.__getURL())
        today = datetime.datetime.now().date()

        self.__validInterval(datetime.date(2008, 1, 1), today)
        self.__logger.info("Total number of intervals: " +
                           str(len(self.__intervals)))
        self.__lastDay = today.strftime("%Y-%m-%d")