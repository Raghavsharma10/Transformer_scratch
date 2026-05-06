def set_copyright(self, copyright_):
        """Sets the copyright.

        arg:    copyright (string): the new copyright
        raise:  InvalidArgument - ``copyright`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``copyright`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetForm.set_title_template
        self._my_map['copyright'] = self._get_display_text(copyright_, self.get_copyright_metadata())