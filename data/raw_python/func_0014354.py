def editor_interfaces(self, space_id, environment_id, content_type_id):
        """
        Provides access to editor interfaces management methods.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/editor-interface

        :return: :class:`EditorInterfacesProxy <contentful_management.editor_interfaces_proxy.EditorInterfacesProxy>` object.
        :rtype: contentful.editor_interfaces_proxy.EditorInterfacesProxy

        Usage:

            >>> editor_interfaces_proxy = client.editor_interfaces('cfexampleapi', 'master', 'cat')
            <EditorInterfacesProxy space_id="cfexampleapi" environment_id="master" content_type_id="cat">
        """

        return EditorInterfacesProxy(self, space_id, environment_id, content_type_id)