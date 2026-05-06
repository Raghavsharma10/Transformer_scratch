def get_assessment_metadata(self):
        """Gets the metadata for an assessment.

        return: (osid.Metadata) - metadata for the assessment
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['assessment'])
        metadata.update({'existing_id_values': self._my_map['assessmentId']})
        return Metadata(**metadata)