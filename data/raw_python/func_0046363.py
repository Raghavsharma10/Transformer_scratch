def get_rubric_metadata(self):
        """Gets the metadata for a rubric assessment.

        return: (osid.Metadata) - metadata for the assesment
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['rubric'])
        metadata.update({'existing_id_values': self._my_map['rubricId']})
        return Metadata(**metadata)