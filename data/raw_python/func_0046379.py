def set_items_shuffled(self, shuffle):
        """Sets the shuffle flag.

        The shuffle flag may be overidden by other assessment sequencing
        rules.

        arg:    shuffle (boolean): ``true`` if the items are shuffled,
                ``false`` if the items appear in the designated order
        raise:  InvalidArgument - ``shuffle`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_group_template
        if self.get_items_shuffled_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_boolean(shuffle):
            raise errors.InvalidArgument()
        self._my_map['itemsShuffled'] = shuffle