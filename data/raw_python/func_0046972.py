def get_assessments_metadata(self):
        """Gets the metadata for the assessments.

        return: (osid.Metadata) - metadata for the assessments
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.get_assets_metadata_template
        metadata = dict(self._mdata['assessments'])
        metadata.update({'existing_assessments_values': self._my_map['assessmentIds']})
        return Metadata(**metadata)