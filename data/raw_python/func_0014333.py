def process(self):
        """
        Calls the process endpoint for all locales of the asset.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/assets/asset-processing
        """

        for locale in self._fields.keys():
            self._client._put(
                "{0}/files/{1}/process".format(
                    self.__class__.base_url(
                        self.space.id,
                        self.id,
                        environment_id=self._environment_id
                    ),
                    locale
                ),
                {},
                headers=self._update_headers()
            )
        return self.reload()