def download(cls, url, filename=None):
        """
        Download a file into the correct cache directory.
        """
        return utility.download(url, cls.directory(), filename)