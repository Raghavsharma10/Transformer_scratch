def clear_published(self):
        """Removes the published status.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_group_template
        if (self.get_published_metadata().is_read_only() or
                self.get_published_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['published'] = self._published_default