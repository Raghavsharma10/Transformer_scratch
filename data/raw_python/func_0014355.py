def snapshots(self, space_id, environment_id, resource_id, resource_kind='entries'):
        """
        Provides access to snapshot management methods.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/snapshots

        :return: :class:`SnapshotsProxy <contentful_management.snapshots_proxy.SnapshotsProxy>` object.
        :rtype: contentful.snapshots_proxy.SnapshotsProxy

        Usage:

            >>> entry_snapshots_proxy = client.snapshots('cfexampleapi', 'master', 'nyancat')
            <SnapshotsProxy[entries] space_id="cfexampleapi" environment_id="master" parent_resource_id="nyancat">

            >>> content_type_snapshots_proxy = client.snapshots('cfexampleapi', 'master', 'cat', 'content_types')
            <SnapshotsProxy[content_types] space_id="cfexampleapi" environment_id="master" parent_resource_id="cat">
        """

        return SnapshotsProxy(self, space_id, environment_id, resource_id, resource_kind)