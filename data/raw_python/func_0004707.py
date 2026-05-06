def update(self, gist, content):
        """
        Updates the contents of file hosted inside a gist at GitHub.
        :param gist: (dict) gist parsed by GitHubTools._parse_gist()
        :param content: (str or bytes) to be written
        :return: (bool) indicatind the success or failure of the update
        """
        # abort if content is False
        if content is False:
            return False

        # request
        url = self._api_url("gists", gist.get("id"))
        data = {"files": {self.filename: {"content": content}}}
        self.output("Sending contents of {} to {}".format(self.file_path, url))
        response = self.requests.patch(url, data=dumps(data))

        # error
        if response.status_code != 200:
            self.oops("Could not update " + gist.get("description"))
            self.oops("PATCH request returned " + str(response.status_code))
            return False

        # success
        self.yeah("Done!")
        self.hey("The URL to this Gist is: {}".format(gist["url"]))
        return True