def snapshots(self):
        """
        Provides access to snapshot management methods for the given entry.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/snapshots

        :return: :class:`EntrySnapshotsProxy <contentful_management.entry_snapshots_proxy.EntrySnapshotsProxy>` object.
        :rtype: contentful.entry_snapshots_proxy.EntrySnapshotsProxy

        Usage:

            >>> entry_snapshots_proxy = entry.snapshots()
            <EntrySnapshotsProxy space_id="cfexampleapi" environment_id="master" entry_id="nyancat">
        """
        return EntrySnapshotsProxy(self._client, self.sys['space'].id, self._environment_id, self.sys['id'])