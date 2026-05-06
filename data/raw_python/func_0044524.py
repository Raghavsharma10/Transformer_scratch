def __validInterval(self, start, finish):
        """Check if the interval is correct.

        An interval is correct if it has less than 1001
        users. If the interval is correct, it will be added
        to '_intervals' attribute. Else, interval will be
        split in two news intervals and these intervals
        will be checked.

        :param start: start date of the interval.
        :type start: datetime.date.
        :param finish: finish date of the interval.
        :type finish: datetime.date.
        """
        url = self.__getURL(1,
                            start.strftime("%Y-%m-%d"),
                            finish.strftime("%Y-%m-%d"))

        data = self.__readAPI(url)

        if data["total_count"] >= 1000:
            middle = start + (finish - start)/2
            self.__validInterval(start, middle)
            self.__validInterval(middle, finish)
        else:
            self.__intervals.append([start.strftime("%Y-%m-%d"),
                                     finish.strftime("%Y-%m-%d")])
            self.__logger.info("New valid interval: " +
                               start.strftime("%Y-%m-%d") +
                               " to " +
                               finish.strftime("%Y-%m-%d"))