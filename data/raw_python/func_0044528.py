def calculeToday(self):
        """Calcule the intervals from the last date."""
        self.__logger.debug("Add today")
        last = datetime.datetime.strptime(self.__lastDay, "%Y-%m-%d")
        today = datetime.datetime.now().date()
        self.__validInterval(last, today)