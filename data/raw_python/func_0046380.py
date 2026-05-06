def clear_items_shuffled(self):
        """Clears the shuffle flag.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_group_template
        if (self.get_items_shuffled_metadata().is_read_only() or
                self.get_items_shuffled_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['itemsShuffled'] = self._items_shuffled_default