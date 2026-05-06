def get_title_metadata(self):
        """Gets the metadata for an asset title.

        return: (osid.Metadata) - metadata for the title
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['title'])
        metadata.update({'existing_string_values': self._my_map['title']})
        return Metadata(**metadata)