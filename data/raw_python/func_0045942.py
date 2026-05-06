def set_composition(self, composition_id):
        """Sets the composition.

        arg:    composition_id (osid.id.Id): a composition
        raise:  InvalidArgument - ``composition_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``composition_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_composition_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(composition_id):
            raise errors.InvalidArgument()
        self._my_map['compositionId'] = str(composition_id)