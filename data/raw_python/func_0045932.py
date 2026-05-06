def get_published_metadata(self):
        """Gets the metadata for the published status.

        return: (osid.Metadata) - metadata for the published field
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['published'])
        metadata.update({'existing_boolean_values': self._my_map['published']})
        return Metadata(**metadata)