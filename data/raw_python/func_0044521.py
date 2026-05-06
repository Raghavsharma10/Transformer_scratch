def __getPeriodUsers(self, start_date, final_date):
        """Get all the users given a period.

        :param start_date: start date of the range to search
            users
        :type start_date: time.date.
        :param final_date: final date of the range to search
            users
        :type final_date: time.date.
        """
        self.__logger.info("Getting users from " + start_date +
                           " to " + final_date)

        url = self.__getURL(1, start_date, final_date)
        data = self.__readAPI(url)
        users = []

        total_pages = 10000
        page = 1

        while total_pages >= page:
            url = self.__getURL(page, start_date, final_date)
            data = self.__readAPI(url)
            self.__logger.debug(str(len(data['items'])) +
                                " users found")
            for u in data['items']:
                users.append(u["login"])
                self.__usersToProccess.put(u["login"])
            total_count = data["total_count"]
            total_pages = int(total_count / 100) + 1
            page += 1
        return users