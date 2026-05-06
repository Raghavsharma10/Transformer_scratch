def clear_score_system(self):
        """Clears the score system.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_avatar_template
        if (self.get_score_system_metadata().is_read_only() or
                self.get_score_system_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['scoreSystemId'] = self._score_system_default