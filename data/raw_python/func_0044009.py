def __getOrganizations(self, web):
        """Scrap the number of organizations from a GitHub profile.

        :param web: parsed web.
        :type web: BeautifulSoup node.
        """
        orgsElements = web.find_all("a", {"class": "avatar-group-item"})
        self.organizations = len(orgsElements)