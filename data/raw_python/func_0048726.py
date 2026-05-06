def add_alt_text(self, alt_text):
        """Adds an alt_text.

        arg:    alt_text (displayText): the new alt_text
        raise:  InvalidArgument - ``alt_text`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``alt_text`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_alt_texts_metadata().is_read_only():
            raise NoAccess()
        self.add_or_replace_value('altTexts', alt_text)