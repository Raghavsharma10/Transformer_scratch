def clear_courses(self):
        """Clears the courses.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.clear_assets_template
        if (self.get_courses_metadata().is_read_only() or
                self.get_courses_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['courseIds'] = self._courses_default