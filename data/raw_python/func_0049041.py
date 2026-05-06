def get_text_metadata(self):
        """Gets the metadata for the text.

        return: (osid.Metadata) - metadata for the text
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['text'])
        metadata.update({'existing_string_values': self._my_map['text']})
        return Metadata(**metadata)