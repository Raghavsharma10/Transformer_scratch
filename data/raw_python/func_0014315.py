def ui_extensions(self):
        """
        Provides access to UI extensions management methods.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/ui-extensions

        :return: :class:`EnvironmentUIExtensionsProxy <contentful_management.ui_extensions_proxy.EnvironmentUIExtensionsProxy>` object.
        :rtype: contentful.ui_extensions_proxy.EnvironmentUIExtensionsProxy

        Usage:

            >>> ui_extensions_proxy = environment.ui_extensions()
            <EnvironmentUIExtensionsProxy space_id="cfexampleapi" environment_id="master">
        """

        return EnvironmentUIExtensionsProxy(self._client, self.space.id, self.id)