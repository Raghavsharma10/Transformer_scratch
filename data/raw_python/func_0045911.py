def clear_copyright(self):
        """Removes the copyright.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetForm.clear_title_template
        if (self.get_copyright_metadata().is_read_only() or
                self.get_copyright_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['copyright'] = dict(self._copyright_default)