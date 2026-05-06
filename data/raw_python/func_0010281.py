def hostname(self):
        """Get the hostname that this connection is associated with"""
        from six.moves.urllib.parse import urlparse
        return urlparse(self._base_url).netloc.split(':', 1)[0]