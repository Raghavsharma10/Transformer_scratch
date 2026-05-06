def clear_text(self):
        """Clears the text.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetForm.clear_title_template
        if (self.get_text_metadata().is_read_only() or
                self.get_text_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['text'] = dict(self._text_default)