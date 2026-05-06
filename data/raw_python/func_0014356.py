def entry_snapshots(self, space_id, environment_id, entry_id):
        """
        Provides access to entry snapshot management methods.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/snapshots

        :return: :class:`SnapshotsProxy <contentful_management.snapshots_proxy.SnapshotsProxy>` object.
        :rtype: contentful.snapshots_proxy.SnapshotsProxy

        Usage:

            >>> entry_snapshots_proxy = client.entry_snapshots('cfexampleapi', 'master', 'nyancat')
            <SnapshotsProxy[entries] space_id="cfexampleapi" environment_id="master" parent_resource_id="nyancat">
        """

        return SnapshotsProxy(self, space_id, environment_id, entry_id, 'entries')