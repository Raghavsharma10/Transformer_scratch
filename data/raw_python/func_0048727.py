def remove_alt_text_language(self, language_type):
        """Removes the specified alt_text.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_alt_texts_metadata().is_read_only():
            raise NoAccess()
        self.remove_field_by_language('altTexts',
                                      language_type)