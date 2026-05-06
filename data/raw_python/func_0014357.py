def content_type_snapshots(self, space_id, environment_id, content_type_id):
        """
        Provides access to content type snapshot management methods.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/snapshots

        :return: :class:`SnapshotsProxy <contentful_management.snapshots_proxy.SnapshotsProxy>` object.
        :rtype: contentful.snapshots_proxy.SnapshotsProxy

        Usage:

            >>> content_type_snapshots_proxy = client.content_type_snapshots('cfexampleapi', 'master', 'cat')
            <SnapshotsProxy[content_types] space_id="cfexampleapi" environment_id="master" parent_resource_id="cat">
        """

        return SnapshotsProxy(self, space_id, environment_id, content_type_id, 'content_types')