def get_grade_system_metadata(self):
        """Gets the metadata for a grading system.

        return: (osid.Metadata) - metadata for the grade system
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['grade_system'])
        metadata.update({'existing_id_values': self._my_map['gradeSystemId']})
        return Metadata(**metadata)