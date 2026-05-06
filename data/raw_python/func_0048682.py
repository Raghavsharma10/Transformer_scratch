def get_group_metadata(self):
        """Gets the metadata for a group.

        return: (osid.Metadata) - metadata for the group
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['group'])
        metadata.update({'existing_boolean_values': self._my_map['group']})
        return Metadata(**metadata)