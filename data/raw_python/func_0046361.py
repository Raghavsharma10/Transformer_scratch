def get_level_metadata(self):
        """Gets the metadata for a grade level.

        return: (osid.Metadata) - metadata for the grade level
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['level'])
        metadata.update({'existing_id_values': self._my_map['levelId']})
        return Metadata(**metadata)