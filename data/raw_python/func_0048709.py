def clear_priority(self):
        """Removes the priority.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.logging.LogEntryForm.clear_priority_template
        if (self.get_priority_metadata().is_read_only() or
                self.get_priority_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['priority'] = self._priority_default