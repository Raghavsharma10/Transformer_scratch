def get_media_descriptions_metadata(self):
        """Gets the metadata for all media descriptions.

        return: (osid.Metadata) - metadata for the media descriptions
        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = dict(self._media_descriptions_metadata)
        metadata.update({'existing_string_values': [t['text'] for t in self.my_osid_object_form._my_map['mediaDescriptions']]})
        return Metadata(**metadata)