def _get_asset(self, asset_uid):
        """
        Returns raw response for an given asset by its unique id.
        """
        uri = self.uri + '/v2/assets/' + asset_uid

        headers = self._get_headers()

        return self.service._get(uri, headers=headers)