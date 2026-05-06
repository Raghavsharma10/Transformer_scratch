def set_score_system(self, grade_system_id):
        """Sets the scoring system.

        arg:    grade_system_id (osid.id.Id): the grade system
        raise:  InvalidArgument - ``grade_system_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_score_system_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(grade_system_id):
            raise errors.InvalidArgument()
        self._my_map['scoreSystemId'] = str(grade_system_id)