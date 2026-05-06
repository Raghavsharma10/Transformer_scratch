def __getURL(self, page=1, start_date=None,
                 final_date=None, order="asc"):
        """Get the API's URL to query to get data about users.

        :param page: number of the page.
        :param start_date: start date of the range to search
            users (Y-m-d).
        "param final_date: final date of the range to search
            users (Y-m-d).
        :param order: order of the query. Valid values are
            'asc' or 'desc'. Default: asc
        :return: formatted URL.
        :rtype: str.
        """
        if not start_date or not final_date:
            url = self.__server + "search/users?client_id=" + \
                self.__githubID + "&client_secret=" + \
                self.__githubSecret + \
                "&order=desc&q=sort:joined+type:user" + \
                self.__urlLocations + \
                self.__urlFilters + \
                "&sort=joined&order=asc&per_page=100&page=" + \
                str(page)
        else:
            url = self.__server + "search/users?client_id=" + \
                self.__githubID + "&client_secret=" + \
                self.__githubSecret + \
                "&order=desc&q=sort:joined+type:user" + \
                self.__urlLocations + \
                self.__urlFilters + \
                "+created:" + \
                start_date + ".." + final_date + \
                "&sort=joined&order=" + order + \
                "&per_page=100&page=" + str(page)
        return url