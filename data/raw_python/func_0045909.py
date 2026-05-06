def get_copyright_metadata(self):
        """Gets the metadata for the copyright.

        return: (osid.Metadata) - metadata for the copyright
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['copyright'])
        metadata.update({'existing_string_values': self._my_map['copyright']})
        return Metadata(**metadata)