def set_cognitive_process(self, grade_id):
        """Sets the cognitive process.

        arg:    grade_id (osid.id.Id): the new cognitive process
        raise:  InvalidArgument - ``grade_id`` is invalid
        raise:  NoAccess - ``grade_id`` cannot be modified
        raise:  NullArgument - ``grade_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_cognitive_process_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(grade_id):
            raise errors.InvalidArgument()
        self._my_map['cognitiveProcessId'] = str(grade_id)