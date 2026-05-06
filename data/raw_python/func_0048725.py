def get_alt_texts_metadata(self):
        """Gets the metadata for all alt_texts.

        return: (osid.Metadata) - metadata for the alt_texts
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._alt_texts_metadata)
        metadata.update({'existing_string_values': [t['text'] for t in self.my_osid_object_form._my_map['altTexts']]})
        return Metadata(**metadata)