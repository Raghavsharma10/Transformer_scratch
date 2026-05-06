def get_cumulative_metadata(self):
        """Gets the metadata for the cumulative flag.

        return: (osid.Metadata) - metadata for the cumulative flag
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['cumulative'])
        metadata.update({'existing_boolean_values': self._my_map['cumulative']})
        return Metadata(**metadata)