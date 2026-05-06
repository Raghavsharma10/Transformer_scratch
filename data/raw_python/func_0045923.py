def get_source_metadata(self):
        """Gets the metadata for the source.

        return: (osid.Metadata) - metadata for the source
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['source'])
        metadata.update({'existing_id_values': self._my_map['sourceId']})
        return Metadata(**metadata)