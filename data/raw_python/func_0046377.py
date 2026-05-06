def clear_items_sequential(self):
        """Clears the items sequential flag.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_group_template
        if (self.get_items_sequential_metadata().is_read_only() or
                self.get_items_sequential_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['itemsSequential'] = self._items_sequential_default