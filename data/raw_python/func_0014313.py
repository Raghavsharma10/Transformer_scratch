def assets(self):
        """
        Provides access to asset management methods.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/assets

        :return: :class:`EnvironmentAssetsProxy <contentful_management.environment_assets_proxy.EnvironmentAssetsProxy>` object.
        :rtype: contentful.environment_assets_proxy.EnvironmentAssetsProxy

        Usage:

            >>> environment_assets_proxy = environment.assets()
            <EnvironmentAssetsProxy space_id="cfexampleapi" environment_id="master">
        """

        return EnvironmentAssetsProxy(self._client, self.space.id, self.id)