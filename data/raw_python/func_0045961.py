def clear_children(self):
        """Clears the children.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_children_metadata().is_read_only() or
                self.get_children_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['childIds'] = self._children_default