def __getBio(self, web):
        """Scrap the bio from a GitHub profile.

        :param web: parsed web.
        :type web: BeautifulSoup node.
        """
        bio = web.find_all("div", {"class": "user-profile-bio"})

        if bio:
            try:
                bio = bio[0].text
                if bio and GitHubUser.isASCII(bio):
                    bioText = bio.replace("\n", "")
                    bioText = bioText.replace("\t", " ").replace("\"", "")
                    bioText = bioText.replace("\'", "").replace("\\", "")
                    self.bio = bioText
                else:
                    self.bio = ""
            except IndexError as error:
                print("There was an error with the user " + self.name)
                print(error)
            except AttributeError as error:
                print("There was an error with the user " + self.name)
                print(error)