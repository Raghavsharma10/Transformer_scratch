def get_taker_metadata(self):
        """Gets the metadata for a resource to manually set which resource will be taking the assessment.

        return: (osid.Metadata) - metadata for the resource
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['taker'])
        metadata.update({'existing_id_values': self._my_map['takerId']})
        return Metadata(**metadata)