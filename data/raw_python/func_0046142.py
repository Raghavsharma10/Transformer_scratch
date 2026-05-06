def get_descriptions_metadata(self):
        """Gets the metadata for all descriptions.

        return: (osid.Metadata) - metadata for the descriptions
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._descriptions_metadata)
        metadata.update({'existing_string_values': [t['text'] for t in self.my_osid_object_form._my_map['descriptions']]})
        return Metadata(**metadata)