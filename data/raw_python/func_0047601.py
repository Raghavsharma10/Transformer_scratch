def get_url(self):
        """Gets the URL associated with this content for web-based retrieval.

        return: (string) - the url for this data
        raise:  IllegalState - ``has_url()`` is ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        # construct the URL from runtime's FILESYSTEM location param
        # plus what we know about the location of repository / assetContents
        # have to get repositoryId from the asset?
        # return self._payload.get_url()
        url = '/repository/repositories/{0}/assets/{1}/contents/{2}/stream'.format(self._my_map['assignedRepositoryIds'][0],
                                                                                   str(self.get_asset_id()),
                                                                                   str(self.get_id()))

        if 'url_hostname' in self._config_map:
            url_hostname = self._config_map['url_hostname']
            return '{0}{1}'.format(url_hostname, url)

        return url