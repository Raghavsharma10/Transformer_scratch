def upload_asset(self, content_type, name, asset):
        """Upload an asset to this release.

        All parameters are required.

        :param str content_type: The content type of the asset. Wikipedia has
            a list of common media types
        :param str name: The name of the file
        :param asset: The file or bytes object to upload.
        :returns: :class:`Asset <Asset>`
        """
        headers = Release.CUSTOM_HEADERS.copy()
        headers.update({'Content-Type': content_type})
        url = self.upload_urlt.expand({'name': name})
        r = self._post(url, data=asset, json=False, headers=headers,
                       verify=False)
        if r.status_code in (201, 202):
            return Asset(r.json(), self)
        raise GitHubError(r)