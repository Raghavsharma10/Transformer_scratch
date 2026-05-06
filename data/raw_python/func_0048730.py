def remove_media_description_language(self, language_type):
        """Removes the specified media_description.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_media_descriptions_metadata().is_read_only():
            raise NoAccess()
        self.remove_field_by_language('mediaDescriptions',
                                      language_type)