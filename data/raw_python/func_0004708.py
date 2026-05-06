def create(self, content, **kwargs):
        """
        Create a new gist.
        :param gist: (dict) gist parsed by GitHubTools._parse()
        :param content: (str or bytes) to be written
        :param public: (bool) defines if the gist is public or private
        :return: (bool) indicatind the success or failure of the creation
        """
        # abort if content is False
        if content is False:
            return False

        # set new gist
        public = bool(kwargs.get("public", True))
        data = {
            "description": self.filename,
            "public": public,
            "files": {self.filename: {"content": content}},
        }

        # send request
        url = self._api_url("gists")
        self.output("Sending contents of {} to {}".format(self.file_path, url))
        response = self.requests.post(url, data=dumps(data))

        # error
        if response.status_code != 201:
            self.oops("Could not create " + self.filename)
            self.oops("POST request returned " + str(response.status_code))
            return False

        # parse created gist
        gist = self._parse_gist(response.json())

        # success
        self.yeah("Done!")
        self.hey("The URL to this Gist is: {}".format(gist["url"]))
        return True