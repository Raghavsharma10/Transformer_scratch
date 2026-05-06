def __getLocation(self, web):
        """Scrap the location from a GitHub profile.

        :param web: parsed web.
        :type web: BeautifulSoup node.
        """
        try:
            self.location = web.find("span", {"class": "p-label"}).text
        except AttributeError as error:
            print("There was an error with the user " + self.name)
            print(error)