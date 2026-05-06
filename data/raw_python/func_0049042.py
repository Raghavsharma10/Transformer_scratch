def set_text(self, text):
        """Sets the text.

        arg:    text (string): the new text
        raise:  InvalidArgument - ``text`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``text`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetForm.set_title_template
        self._my_map['text'] = self._get_display_text(text, self.get_text_metadata())