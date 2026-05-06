def entries(self):
        """
        Provides access to entry management methods for the given content type.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/entries

        :return: :class:`ContentTypeEntriesProxy <contentful_management.content_type_entries_proxy.ContentTypeEntriesProxy>` object.
        :rtype: contentful.content_type_entries_proxy.ContentTypeEntriesProxy

        Usage:

            >>> content_type_entries_proxy = content_type.entries()
            <ContentTypeEntriesProxy space_id="cfexampleapi" environment_id="master" content_type_id="cat">
        """
        return ContentTypeEntriesProxy(self._client, self.space.id, self._environment_id, self.id)