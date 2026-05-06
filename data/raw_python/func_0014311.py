def content_types(self):
        """
        Provides access to content type management methods for content types of an environment.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/content-types

        :return: :class:`EnvironmentContentTypesProxy <contentful_management.space_content_types_proxy.EnvironmentContentTypesProxy>` object.
        :rtype: contentful.space_content_types_proxy.EnvironmentContentTypesProxy

        Usage:

            >>> space_content_types_proxy = environment.content_types()
            <EnvironmentContentTypesProxy space_id="cfexampleapi" environment_id="master">
        """

        return EnvironmentContentTypesProxy(self._client, self.space.id, self.id)