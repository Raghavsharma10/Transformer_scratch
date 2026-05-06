def join(self, path):
        """
        Similar to :func:`os.path.join` but returns a storage object instead.

        :param str path: path to join on to this object's URI
        :returns: a storage object
        :rtype: BaseURI
        """

        return self.parse_uri(urlparse(os.path.join(str(self), path)), storage_args=self.storage_args)