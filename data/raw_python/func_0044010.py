def getData(self):
        """Get data of the GitHub user."""
        url = self.server + self.name
        data = GitHubUser.__getDataFromURL(url)
        web = BeautifulSoup(data, "lxml")
        self.__getContributions(web)
        self.__getLocation(web)
        self.__getAvatar(web)
        self.__getNumberOfRepositories(web)
        self.__getNumberOfFollowers(web)
        self.__getBio(web)
        self.__getJoin(web)
        self.__getOrganizations(web)