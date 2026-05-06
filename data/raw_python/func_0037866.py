def _get_shards(self):
        """
        Returns comma separated list of configured Solr cores
        """
        if self._shards is None:
            endpoints = []
            for endpoint in self.endpoints:
                # We need to remove and http:// prefixes from URLs
                url = urlparse.urlparse(self.endpoints[endpoint])
                endpoints.append("/".join([url.netloc, url.path]))
            self._shards = ",".join(endpoints)
        return self._shards