def clear_grade_system(self):
        """Clears the grading system.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_avatar_template
        if (self.get_grade_system_metadata().is_read_only() or
                self.get_grade_system_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['gradeSystemId'] = self._grade_system_default