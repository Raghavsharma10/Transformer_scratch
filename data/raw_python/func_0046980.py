def clear_level(self):
        """Clears the level.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_level_metadata().is_read_only() or
                self.get_level_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['levelId'] = self._level_default
        self._my_map['level'] = self._level_default