def call(self):
        """ Make the API call again and fetch fresh data. """
        data = self._downloader.download()

        # Only try to pass errors arg if supported
        if sys.version >= "2.7":
            data = data.decode("utf-8", errors="ignore")
        else:
            data = data.decode("utf-8")

        self.update(json.loads(data))
        self._fetched = True