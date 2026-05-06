def get_composition_metadata(self):
        """Gets the metadata for linking this asset to a composition.

        return: (osid.Metadata) - metadata for the composition
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['composition'])
        metadata.update({'existing_id_values': self._my_map['compositionId']})
        return Metadata(**metadata)