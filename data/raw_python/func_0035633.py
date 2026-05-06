def searchPlanet(self, name):
        """ Searches the database for a planet. Input can be complete ie GJ1214b, alternate name variations or even
        just 1214.

        :param name: the name of the planet to search
        :return: dictionary of results as planetname -> planet object
        """

        searchName = compactString(name)
        returnDict = {}

        for altname, planetObj in self._planetSearchDict.iteritems():
            if re.search(searchName, altname):
                returnDict[planetObj.name] = planetObj

        if returnDict:
            if len(returnDict) == 1:
                return returnDict.values()[0]
            else:
                return returnDict.values()

        else:
            return False