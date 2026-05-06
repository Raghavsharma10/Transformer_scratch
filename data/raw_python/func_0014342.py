def editor_interfaces(self):
        """
        Provides access to editor interface management methods for the given content type.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/editor-interface

        :return: :class:`ContentTypeEditorInterfacesProxy <contentful_management.content_type_editor_interfaces_proxy.ContentTypeEditorInterfacesProxy>` object.
        :rtype: contentful.content_type_editor_interfaces_proxy.ContentTypeEditorInterfacesProxy

        Usage:

            >>> content_type_editor_interfaces_proxy = content_type.editor_interfaces()
            <ContentTypeEditorInterfacesProxy space_id="cfexampleapi" environment_id="master" content_type_id="cat">
        """
        return ContentTypeEditorInterfacesProxy(self._client, self.space.id, self._environment_id, self.id)