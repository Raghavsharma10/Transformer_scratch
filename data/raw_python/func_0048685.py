def get_avatar_metadata(self):
        """Gets the metadata for an asset.

        return: (osid.Metadata) - metadata for the asset
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['avatar'])
        metadata.update({'existing_id_values': self._my_map['avatarId']})
        return Metadata(**metadata)