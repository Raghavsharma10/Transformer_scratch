def get_duration_metadata(self):
        """Gets the metadata for the assessment duration.

        return: (osid.Metadata) - metadata for the duration
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['duration'])
        metadata.update({'existing_duration_values': self._my_map['duration']})
        return Metadata(**metadata)