def valid(self):
        """Validates of WebDAV and proxy settings.

        :return: True in case settings are valid and False otherwise.
        """
        return True if self.webdav.valid() and self.proxy.valid() else False