def snapshots(self):
        """
        Provides access to snapshot management methods for the given content type.

        API reference: https://www.contentful.com/developers/docs/references/content-management-api/#/reference/snapshots/content-type-snapshots-collection

        :return: :class:`ContentTypeSnapshotsProxy <contentful_management.content_type_snapshots_proxy.ContentTypeSnapshotsProxy>` object.
        :rtype: contentful.content_type_snapshots_proxy.ContentTypeSnapshotsProxy

        Usage:

            >>> content_type_snapshots_proxy = content_type.entries()
            <ContentTypeSnapshotsProxy space_id="cfexampleapi" environment_id="master" content_type_id="cat">
        """
        return ContentTypeSnapshotsProxy(self._client, self.space.id, self._environment_id, self.id)