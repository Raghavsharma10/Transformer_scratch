def __getNumberOfRepositories(self, web):
        """Scrap the number of repositories from a GitHub profile.

        :param web: parsed web.
        :type web: BeautifulSoup node.
        """
        counters = web.find_all('span', {'class': 'Counter'})
        try:
            if 'k' not in counters[0].text:
                self.numberOfRepos = int(counters[0].text)
            else:
                reposText = counters[0].text.replace(" ", "")
                reposText = reposText.replace("\n", "").replace("k", "")

                if reposText and len(reposText) > 1:
                    self.numberOfRepos = int(reposText.split(".")[0]) * \
                        1000 + int(reposText.split(".")[1]) * 100
                elif reposText:
                    self.numberOfRepos = int(reposText.split(".")[0]) * 1000
        except IndexError as error:
            print("There was an error with the user " + self.name)
            print(error)
        except AttributeError as error:
            print("There was an error with the user " + self.name)
            print(error)