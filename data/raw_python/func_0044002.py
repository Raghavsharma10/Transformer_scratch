def __getContributions(self, web):
        """Scrap the contributions from a GitHub profile.

        :param web: parsed web.
        :type web: BeautifulSoup node.
        """
        contributions_raw = web.find_all('h2',
                                         {'class': 'f4 text-normal mb-2'})
        try:
            contrText = contributions_raw[0].text
            contrText = contrText.lstrip().split(" ")[0]
            contrText = contrText.replace(",", "")
        except IndexError as error:
            print("There was an error with the user " + self.name)
            print(error)
        except AttributeError as error:
            print("There was an error with the user " + self.name)
            print(error)

        self.contributions = int(contrText)