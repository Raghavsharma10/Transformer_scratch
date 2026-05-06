def clear_title(self):
        """Removes the title.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetForm.clear_title_template
        if (self.get_title_metadata().is_read_only() or
                self.get_title_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['title'] = dict(self._title_default)