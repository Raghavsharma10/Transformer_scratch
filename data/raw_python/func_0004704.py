def get_gists(self):
        """
        List generator containing gist relevant information
        such as id, description, filenames and raw URL (dict).
        """
        # fetch all gists
        if self.is_authenticated:
            url = self._api_url("gists")
        else:
            url = self._api_url("users", self.user, "gists")
        self.output("Fetching " + url)
        raw_resp = self.requests.get(url)

        # abort if user not found
        if raw_resp.status_code != 200:
            self.oops("User `{}` not found".format(self.user))
            return

        # abort if there are no gists
        resp = raw_resp.json()
        if not resp:
            self.oops("No gists found for user `{}`".format(self.user))
            return

        # parse response
        for gist in raw_resp.json():
            yield self._parse_gist(gist)