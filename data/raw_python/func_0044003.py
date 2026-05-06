def __getAvatar(self, web):
        """Scrap the avatar from a GitHub profile.

        :param web: parsed web.
        :type web: BeautifulSoup node.
        """
        try:
            self.avatar = web.find("img", {"class": "avatar"})['src'][:-10]
        except IndexError as error:
            print("There was an error with the user " + self.name)
            print(error)
        except AttributeError as error:
            print("There was an error with the user " + self.name)
            print(error)