def set_priority(self, priority):
        """Sets the priority.

        arg:    priority (osid.type.Type): the new priority
        raise:  InvalidArgument - ``priority`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``priority`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.logging.LogEntryForm.set_priority
        if self.get_priority_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_type(priority):
            raise errors.InvalidArgument()
        self._my_map['priority'] = str(priority)