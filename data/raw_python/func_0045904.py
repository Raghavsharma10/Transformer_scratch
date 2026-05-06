def set_title(self, title):
        """Sets the title.

        arg:    title (string): the new title
        raise:  InvalidArgument - ``title`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``title`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetForm.set_title_template
        self._my_map['title'] = self._get_display_text(title, self.get_title_metadata())