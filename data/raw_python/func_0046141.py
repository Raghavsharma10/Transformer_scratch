def get_display_names_metadata(self):
        """Gets the metadata for all display_names.

        return: (osid.Metadata) - metadata for the display_names
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._display_names_metadata)
        metadata.update({'existing_string_values': [t['text'] for t in self.my_osid_object_form._my_map['displayNames']]})
        return Metadata(**metadata)