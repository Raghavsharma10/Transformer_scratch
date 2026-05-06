def locales(self):
        """
        Provides access to locale management methods.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/locales

        :return: :class:`EnvironmentLocalesProxy <contentful_management.environment_locales_proxy.EnvironmentLocalesProxy>` object.
        :rtype: contentful.environment_locales_proxy.EnvironmentLocalesProxy

        Usage:

            >>> environment_locales_proxy = environment.locales()
            <EnvironmentLocalesProxy space_id="cfexampleapi" environment_id="master">
        """

        return EnvironmentLocalesProxy(self._client, self.space.id, self.id)