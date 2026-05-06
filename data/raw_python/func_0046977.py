def get_completion_metadata(self):
        """Gets the metadata for completion percentage.

        return: (osid.Metadata) - metadata for the completion percentage
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['completion'])
        metadata.update({'existing_decimal_values': self._my_map['completion']})
        return Metadata(**metadata)