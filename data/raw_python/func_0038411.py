def get_download_link(self):
        """ Get direct download link with soudcloud's redirect system. """
        url = None
        if not self.get("downloadable"):
            try:
                url = self.client.get_location(
                    self.client.STREAM_URL % self.get("id"))
            except serror as e:
                print(e)

        if not url:
            try:
                url = self.client.get_location(
                    self.client.DOWNLOAD_URL % self.get("id"))
            except serror as e:
                print(e)

        return url