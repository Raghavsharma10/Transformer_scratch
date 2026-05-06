def get_assets_metadata(self):
        """Gets the metadata for the assets.

        return: (osid.Metadata) - metadata for the assets
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.get_assets_metadata_template
        metadata = dict(self._mdata['assets'])
        metadata.update({'existing_assets_values': self._my_map['assetIds']})
        return Metadata(**metadata)