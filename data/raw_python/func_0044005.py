def __getNumberOfFollowers(self, web):
        """Scrap the number of followers from a GitHub profile.

        :param web: parsed web.
        :type web: BeautifulSoup node.
        """
        counters = web.find_all('span', {'class': 'Counter'})
        try:
            if 'k' not in counters[2].text:
                self.followers = int(counters[2].text)
            else:
                follText = counters[2].text.replace(" ", "")
                follText = follText.replace("\n", "").replace("k", "")

                if follText and len(follText) > 1:
                    self.followers = int(follText.split(".")[0])*1000 + \
                        int(follText.split(".")[1]) * 100
                elif follText:
                    self.followers = int(follText.split(".")[0])*1000
        except IndexError as error:
            print("There was an error with the user " + self.name)
            print(error)
        except AttributeError as error:
            print("There was an error with the user " + self.name)
            print(error)