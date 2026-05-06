def download(self, source, destination):
        """Downloads the file on the disk at source into destination."""
        self._handler.download(posix_path(source), destination)