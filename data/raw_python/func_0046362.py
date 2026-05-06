def set_level(self, grade_id):
        """Sets the level of difficulty expressed as a ``Grade``.

        arg:    grade_id (osid.id.Id): the grade level
        raise:  InvalidArgument - ``grade_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``grade_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_level_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(grade_id):
            raise errors.InvalidArgument()
        self._my_map['levelId'] = str(grade_id)