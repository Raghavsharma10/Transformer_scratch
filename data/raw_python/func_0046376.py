def set_items_sequential(self, sequential):
        """Sets the items sequential flag.

        arg:    sequential (boolean): ``true`` if the items are taken
                sequentially, ``false`` if the items can be skipped and
                revisited
        raise:  InvalidArgument - ``sequential`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_items_sequential_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(sequential):
            raise errors.InvalidArgument()
        self._my_map['itemsSequential'] = sequential