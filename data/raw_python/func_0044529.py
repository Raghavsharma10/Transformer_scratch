def __addLocationsToURL(self, locations):
        """Format all locations to GitHub's URL API.

        :param locations: locations where to search users.
        :type locations: list(str).
        """
        for l in self.__locations:
            self.__urlLocations += "+location:\""\
             + str(quote(l)) + "\""