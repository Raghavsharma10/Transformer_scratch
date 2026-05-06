def _get_profile(self):
        """
        Retrieves the profile data of the user and formats it as a
        Python dictionary.
        """
        url = PROFILE_URL + self.hash + '.json'
        try:
            profile = json.load(urlopen(url))
            # set the profile as an instance variable
            self._profile = profile['entry'][0]
        except:
            self._profile = {}