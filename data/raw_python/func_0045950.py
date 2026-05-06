def get_data_metadata(self):
        """Gets the metadata for the content data.

        return: (osid.Metadata) - metadata for the content data
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['data'])
        metadata.update({'existing_object_values': self._my_map['data']})
        return Metadata(**metadata)