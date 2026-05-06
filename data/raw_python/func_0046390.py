def get_score_system_metadata(self):
        """Gets the metadata for a score system.

        return: (osid.Metadata) - metadata for the grade system
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['score_system'])
        metadata.update({'existing_id_values': self._my_map['scoreSystemId']})
        return Metadata(**metadata)