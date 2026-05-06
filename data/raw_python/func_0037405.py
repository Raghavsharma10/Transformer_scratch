def _update_metadata(self, archive_name, archive_metadata):
        """
        Appends the updated_metada dict to the Metadata Attribute list

        Parameters
        ----------
        archive_name: str

            ID of archive to update

        updated_metadata: dict

            dictionary of metadata keys and values to update. If the value
            for a particular key is `None`, the key is removed.

        """

        archive_metadata_current = self._get_archive_metadata(archive_name)
        archive_metadata_current.update(archive_metadata)
        for k, v in archive_metadata_current.items():
            if v is None:
                del archive_metadata_current[k]

        # add the updated archive_metadata object to Dynamo
        self._table.update_item(
            Key={'_id': archive_name},
            UpdateExpression="SET archive_metadata = :v",
            ExpressionAttributeValues={':v': archive_metadata_current},
            ReturnValues='ALL_NEW')